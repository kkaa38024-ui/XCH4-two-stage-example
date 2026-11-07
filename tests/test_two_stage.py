"""Synthetic fixtures test software contracts, never satellite observations."""
import unittest
import tempfile
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
from two_stage import (prepare, fit_stage1, fit_stage2, training_frame,
    temporal_holdout, validation_splits, spatial_blocks, cross_validate,
    export_netcdf, load_model, metrics)


def synthetic_fixture(n=36):
    rng = np.random.default_rng(7)
    f = pd.DataFrame({c: rng.normal(size=n) for c in
                      ['t2m','tp','v10','u10','ssr','sp','NDVI','LST','DEM']})
    f['time'] = pd.date_range('2019-01-01', periods=n, freq='MS')
    f['lat'] = np.linspace(19, 53, n)
    f['lon'] = np.linspace(74, 134, n)
    f['row_id'] = np.arange(n) + 100
    f['XCH4_pred'] = 1800 + rng.normal(size=n)
    f['XCH4_go'] = f.XCH4_pred + 4 + f.t2m
    f['XCH4_tr'] = 1800 + f.t2m
    return f


class Contracts(unittest.TestCase):
    def test_archived_preset_seasons_and_serialization(self):
        f = synthetic_fixture(60)
        fitted = fit_stage2(f.iloc[:48], preset='archived_loyo')
        self.assertEqual(fitted.model.n_estimators, 800)
        self.assertEqual(len(fitted.features), 13)
        out = fitted.predict(f.iloc[48:])
        np.testing.assert_array_equal(out.t_index, np.arange(48, 60))
        np.testing.assert_allclose(out.month_sin, np.sin(2*np.pi*np.arange(1,13)/12))
        np.testing.assert_allclose(out.month_cos, np.cos(2*np.pi*np.arange(1,13)/12))
        with tempfile.TemporaryDirectory() as tmp:
            fitted.save(Path(tmp)/'model')
            loaded = load_model(Path(tmp)/'model')
            self.assertEqual(loaded.preset, 'archived_loyo')
            self.assertEqual(loaded.features, fitted.features)
            np.testing.assert_array_equal(loaded.predict(f).XCH4_bc, fitted.predict(f).XCH4_bc)

    def test_pso_synthetic_internal_split_reproducible(self):
        from two_stage import tune_stage1
        from unittest.mock import patch
        from sklearn.model_selection import train_test_split
        f = synthetic_fixture(24)
        internal, _ = train_test_split(np.arange(len(f)), train_size=.6, random_state=42)
        original_fit = __import__('two_stage').MinMaxScaler.fit
        seen = []
        def track(scaler, x, *args, **kwargs):
            seen.append(x.copy())
            return original_fit(scaler, x, *args, **kwargs)
        with patch('two_stage.MinMaxScaler.fit', track):
            first = tune_stage1(f, swarmsize=2, maxiter=1)
            second = tune_stage1(f, swarmsize=2, maxiter=1)
        self.assertEqual(first, second)
        self.assertEqual(len(seen), 2)
        for x in seen:
            np.testing.assert_array_equal(x.t2m, f.iloc[internal].t2m)
        for key, lower, upper in [('n_estimators',200,800),('max_depth',4,10),
                                  ('learning_rate',.01,.2),('subsample',.6,1),('colsample_bytree',.6,1)]:
            self.assertTrue(lower <= first[key] <= upper)
        self.assertEqual(first['n_jobs'], 2)

    def test_missing_stage1_label_is_not_fabricated(self):
        with self.assertRaisesRegex(ValueError, 'XCH4_tr'):
            fit_stage1(synthetic_fixture().drop(columns='XCH4_tr'), params={'n_estimators': 2})

    def test_fixed_time_index_subset(self):
        f = synthetic_fixture(60)
        subset = prepare(f.iloc[48:])
        self.assertEqual(subset.t_index.tolist(), list(range(48, 60)))
        fitted = fit_stage2(f.iloc[:48], params={'n_estimators': 2})
        self.assertEqual(fitted.predict(f.iloc[48:]).t_index.tolist(), list(range(48, 60)))

    def test_nan_labels_and_predictors(self):
        f = synthetic_fixture()
        f.loc[0, 'XCH4_tr'] = np.nan
        f.loc[1, 'NDVI'] = np.nan
        valid, _ = training_frame(f, 1)
        self.assertEqual(len(valid), len(f)-1)
        self.assertTrue(valid.NDVI.isna().any())
        fitted = fit_stage1(f, params={'n_estimators': 3})
        self.assertEqual(len(fitted.predict(f)), len(f))
        self.assertTrue(np.isfinite(fitted.predict(f).XCH4_pred).all())
        f['XCH4_tr'] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite valid targets'):
            fit_stage1(f, params={'n_estimators': 2})

    def test_validation_errors(self):
        f = synthetic_fixture()
        for column, value in [('time', 'bad-date'), ('lat', np.nan), ('t2m', np.inf),
                              ('XCH4_go', np.inf), ('tp', 'not numeric')]:
            bad = f.copy()
            bad[column] = value
            with self.assertRaises((ValueError, TypeError)):
                prepare(bad)
        with self.assertRaises(ValueError):
            prepare(f.drop(columns='LST'))
        f.loc[1, 'row_id'] = f.loc[0, 'row_id']
        with self.assertRaisesRegex(ValueError, 'row_id'):
            prepare(f)

    def test_group_disjoint_exact_folds_and_reproducibility(self):
        f = prepare(synthetic_fixture(60).sample(frac=1, random_state=11))
        for kind in ['sample', 'spatial', 'month']:
            splits = validation_splits(f, kind, 5)
            self.assertEqual(len(splits), 5)
            self.assertEqual(sorted(np.concatenate([te for _, te in splits]).tolist()), list(range(len(f))))
            for (tr, te), (tr2, te2) in zip(splits, validation_splits(f, kind, 5)):
                np.testing.assert_array_equal(te, te2)
                self.assertFalse(set(f.iloc[tr].row_id) & set(f.iloc[te].row_id))
                if kind != 'sample':
                    g = spatial_blocks(f) if kind == 'spatial' else f.time.dt.to_period('M').to_numpy()
                    self.assertFalse(set(g[tr]) & set(g[te]))
        with self.assertRaisesRegex(ValueError, 'Fewer groups'):
            validation_splits(f.iloc[:2], 'month', 3)
        with self.assertRaises(ValueError):
            validation_splits(f, 'sample', 100)

    def test_shuffle_alignment_and_cv(self):
        f = synthetic_fixture().sample(frac=1, random_state=19)
        out = cross_validate(f, stage=2, kind='month', n_splits=3, params={'n_estimators': 2})
        source = f.set_index('row_id')
        self.assertEqual(set(out.row_id), set(source.index))
        for row in out.itertuples():
            self.assertEqual(row.XCH4_go, source.loc[row.row_id, 'XCH4_go'])
            self.assertEqual(row.t2m, source.loc[row.row_id, 't2m'])
        self.assertEqual(out.fold.nunique(), 3)

    def test_spatial_domain_fixed_edges(self):
        f = synthetic_fixture(2)
        f['lat'] = [18.35, 53.45]
        f['lon'] = [73.95, 134.85]
        np.testing.assert_array_equal(spatial_blocks(f), [0, 99])
        self.assertEqual(spatial_blocks(f.iloc[1:])[0], 99)
        f.loc[0, 'lat'] = 18
        with self.assertRaises(ValueError):
            spatial_blocks(f)

    def test_holdout_no_residual_exclusion_or_target_leakage(self):
        f = synthetic_fixture(60)
        f.loc[59, 'XCH4_go'] = 100000  # Extreme synthetic residual must stay.
        fitted, out, scores = temporal_holdout(f)
        self.assertEqual(len(out), 12)
        self.assertTrue(out.evaluation_eligible.all())
        self.assertEqual(scores['baseline']['n'], scores['corrected']['n'])
        self.assertEqual(scores['corrected']['n'], 12)
        self.assertIn(100000, out.XCH4_go.to_list())
        self.assertEqual(fitted.scaler.data_max_[-1], 47)  # Training-only scaling.
        changed = f.copy()
        changed.loc[48:, 'XCH4_go'] = -10000
        _, other, _ = temporal_holdout(changed)
        np.testing.assert_array_equal(out.XCH4_bc, other.XCH4_bc)
        pd.testing.assert_frame_equal(fitted.predict(f.iloc[48:]).drop(columns='XCH4_go'),
                                      fitted.predict(changed.iloc[48:]).drop(columns='XCH4_go'))

    def test_holdout_missing_labels_marked_not_imputed(self):
        f = synthetic_fixture(60)
        f.loc[48, 'XCH4_go'] = np.nan
        f.loc[49, 'XCH4_pred'] = np.nan
        _, out, scores = temporal_holdout(f)
        self.assertEqual(len(out), 12)
        self.assertEqual(scores['baseline']['n'], 10)
        self.assertTrue(np.isnan(out.loc[out.row_id.eq(149), 'XCH4_bc'].iloc[0]))

    def test_save_load_csv_and_sparse_netcdf(self):
        f = synthetic_fixture(60)
        fitted, out, _ = temporal_holdout(f)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fitted.save(path/'model.joblib')
            loaded = load_model(path/'model.joblib')
            np.testing.assert_array_equal(fitted.predict(f).XCH4_bc, loaded.predict(f).XCH4_bc)
            out.to_csv(path/'pred.csv', index=False)
            csv = pd.read_csv(path/'pred.csv')
            export_netcdf(csv, path/'pred.nc')
            with xr.open_dataset(path/'pred.nc') as ds:
                self.assertEqual(dict(ds.sizes), {'sample': 12})
                np.testing.assert_array_equal(ds.row_id.values, out.row_id)
                np.testing.assert_allclose(ds.XCH4_bc.values, out.XCH4_bc)
                self.assertEqual(ds.XCH4_bc.attrs['units'], 'ppb')
                np.testing.assert_array_equal(ds.time.values, out.time.to_numpy())

    def test_metrics_refuse_implicit_filter(self):
        with self.assertRaises(ValueError):
            metrics([1, 2], [1, np.nan])

if __name__ == '__main__':
    unittest.main()
