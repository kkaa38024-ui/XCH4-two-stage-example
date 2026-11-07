"""Compact two-stage XCH4 example with explicit source presets and optional PSO."""
from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

FEATURES = ['t2m', 'tp', 'v10', 'u10', 'ssr', 'sp', 'NDVI', 'LST', 'DEM', 't_index']
STAGE2_FEATURES = ['XCH4_pred'] + FEATURES
STAGE2_PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8)
STAGE2_PRESETS = {
    'primary': (STAGE2_FEATURES, STAGE2_PARAMS),
    'archived_loyo': (STAGE2_FEATURES + ['month_sin', 'month_cos'],
                      dict(n_estimators=800, max_depth=8, learning_rate=0.02,
                           subsample=0.8, colsample_bytree=0.9)),
}
STUDY_BOUNDS = (18.35, 53.45, 73.95, 134.85)


def prepare(frame, base_year=2019):
    """Keep input order; row_id is an explicit unique record key, not a grid key."""
    required = ['time', 'lat', 'lon'] + FEATURES[:-1]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    f = frame.copy()
    if 'row_id' not in f:
        f['row_id'] = np.arange(len(f), dtype=np.int64)
    if f.row_id.isna().any() or f.row_id.duplicated().any():
        raise ValueError('row_id must be nonmissing and unique')
    f['time'] = pd.to_datetime(f.time, errors='raise', format='mixed')
    if f.time.isna().any() or f.time.dt.tz is not None:
        raise ValueError('time must contain valid timezone-naive dates')
    for col in required[1:] + [c for c in ['XCH4_tr', 'XCH4_go', 'XCH4_pred'] if c in f]:
        try:
            f[col] = pd.to_numeric(f[col], errors='raise').astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{col} must be numeric') from exc
        if np.isinf(f[col].to_numpy()).any():
            raise ValueError(f'{col} contains infinity')
    if not np.isfinite(f[['lat', 'lon']].to_numpy()).all():
        raise ValueError('lat/lon must be finite')
    if ((f.lat.abs() > 90) | (f.lon.abs() > 180)).any():
        raise ValueError('Invalid geographic coordinates')
    f['t_index'] = (f.time.dt.year - base_year) * 12 + f.time.dt.month - 1
    f['month_sin'] = np.sin(2 * np.pi * f.time.dt.month / 12)
    f['month_cos'] = np.cos(2 * np.pi * f.time.dt.month / 12)
    return f.reset_index(drop=True)


def training_frame(frame, stage, base_year=2019):
    f = prepare(frame, base_year)
    label = 'XCH4_tr' if stage == 1 else 'XCH4_go'
    needed = [label] + (['XCH4_pred'] if stage == 2 else [])
    if any(c not in f for c in needed):
        raise ValueError(f'Missing training columns: {needed}; observed labels are required')
    valid = np.isfinite(f[needed].to_numpy()).all(axis=1)
    f = f.loc[valid].reset_index(drop=True)
    if len(f) < 2:
        raise ValueError('At least two finite valid targets are required')
    y = f[label].to_numpy()
    if stage == 2:
        y = y - f.XCH4_pred.to_numpy()
    return f, y


def estimator(params):
    p = dict(objective='reg:squarederror', tree_method='hist', random_state=42, n_jobs=2)
    p.update(params)
    if p['n_jobs'] not in (1, 2):
        raise ValueError('n_jobs must be bounded to 1 or 2')
    return XGBRegressor(**p)


@dataclass
class FittedStage:
    stage: int
    base_year: int
    scaler: MinMaxScaler
    model: XGBRegressor
    preset: str = 'primary'
    features: list = None

    def __post_init__(self):
        if self.features is None:
            self.features = list(FEATURES if self.stage == 1 else STAGE2_PRESETS[self.preset][0])

    def predict(self, frame):
        f = prepare(frame, self.base_year)
        if self.stage == 2 and 'XCH4_pred' not in f:
            raise ValueError('Missing XCH4_pred Mapping baseline')
        delta_or_mapping = self.model.predict(self.scaler.transform(f[self.features]))
        if self.stage == 1:
            f['XCH4_pred'] = delta_or_mapping
        else:
            f['predicted_delta'] = delta_or_mapping
            f['XCH4_bc'] = f.XCH4_pred + delta_or_mapping
            # A missing baseline cannot produce a corrected concentration.
            f.loc[f.XCH4_pred.isna(), ['predicted_delta', 'XCH4_bc']] = np.nan
        return f

    def save(self, path):
        """Local trusted artifacts only; never load untrusted pickle/joblib files."""
        joblib.dump(dict(stage=self.stage, base_year=self.base_year,
                         scaler=self.scaler, model=self.model,
                         preset=self.preset, features=self.features), path)


def load_model(path):
    return FittedStage(**joblib.load(path))


def _fit(frame, stage, params, base_year, preset='primary'):
    f, y = training_frame(frame, stage, base_year)
    features = list(FEATURES if stage == 1 else STAGE2_PRESETS[preset][0])
    scaler = MinMaxScaler()
    x = scaler.fit_transform(f[features])  # NaNs remain NaNs for XGBoost.
    model = estimator(params)
    model.fit(x, y)
    return FittedStage(stage, base_year, scaler, model, preset, features)


def fit_stage1(frame, *, params, base_year=2019):
    """Requires actual observed XCH4_tr and explicit author-selected parameters."""
    return _fit(frame, 1, params, base_year)


def fit_stage2(frame, *, params=None, base_year=2019, preset='primary'):
    if preset not in STAGE2_PRESETS:
        raise ValueError(f'Unknown Stage2 preset: {preset}')
    selected_params = dict(STAGE2_PRESETS[preset][1])
    if params is not None:
        selected_params.update(params)
    return _fit(frame, 2, selected_params, base_year, preset)


def tune_stage1(frame, *, base_year=2019, seed=42, swarmsize=16, maxiter=8):
    """Primary source PSO using only an internal 60/40 development split.

    Supply author training data only. Full refit is a separate fit_stage1 call.
    The optional pyswarm dependency uses its 0.6-compatible public interface.
    """
    from pyswarm import pso
    from inspect import signature
    f, y = training_frame(frame, 1, base_year)
    train, validation = train_test_split(np.arange(len(f)), train_size=.6, random_state=seed)
    if len(validation) < 2:
        raise ValueError('Internal validation requires at least two rows')
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(f.iloc[train][FEATURES])
    x_validation = scaler.transform(f.iloc[validation][FEATURES])
    names = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']

    def parameters(values):
        params = dict(zip(names, values))
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params.update(n_jobs=2, random_state=seed)
        return params

    def objective(values):
        model = estimator(parameters(values))
        model.fit(x_train, y[train])
        return -r2_score(y[validation], model.predict(x_validation))

    # pyswarm 0.6 uses NumPy's global RNG. Restore caller state after tuning.
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        options = dict(swarmsize=swarmsize, maxiter=maxiter)
        if 'seed' in signature(pso).parameters:
            options['seed'] = seed
        result = pso(objective, [200, 4, .01, .6, .6], [800, 10, .2, 1., 1.], **options)
        best = result.x if hasattr(result, 'x') else result[0]
    finally:
        np.random.set_state(state)
    return parameters(best)


def spatial_blocks(frame, n_lat=10, n_lon=10, bounds=STUDY_BOUNDS):
    """Fixed study-domain bins; include both outer edges, reject outside points."""
    south, north, west, east = bounds
    if n_lat < 1 or n_lon < 1 or south >= north or west >= east:
        raise ValueError('Invalid block configuration')
    a = frame[['lat', 'lon']].to_numpy(dtype=float)
    if not np.isfinite(a).all() or ((a < [south, west]) | (a > [north, east])).any():
        raise ValueError('Coordinates outside explicit study bounds')
    bins = np.floor((a - [south, west]) / [north-south, east-west] * [n_lat, n_lon]).astype(int)
    bins = np.minimum(bins, [n_lat-1, n_lon-1])
    return bins[:, 0] * n_lon + bins[:, 1]


def validation_splits(frame, kind='sample', n_splits=10, seed=42):
    """Positions in this finalized frame. Group folds shuffle groups, not rows.

    Month-group CV tests unseen months, not strictly forward forecasting.
    """
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError('n_splits must be an integer >= 2')
    if kind == 'sample':
        if len(frame) < n_splits:
            raise ValueError('Fewer samples than requested n_splits')
        return list(KFold(n_splits, shuffle=True, random_state=seed).split(frame))
    if kind == 'spatial':
        groups = spatial_blocks(frame)
    elif kind == 'month':
        groups = frame.time.dt.to_period('M').astype(str).to_numpy()
    else:
        raise ValueError('kind must be sample, spatial, or month')
    unique = np.unique(groups)
    if len(unique) < n_splits:
        raise ValueError('Fewer groups than requested n_splits')
    unique = np.random.default_rng(seed).permutation(unique)
    folds = []
    for held_groups in np.array_split(unique, n_splits):
        test = np.isin(groups, held_groups)
        folds.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return folds


def cross_validate(frame, *, stage, kind='sample', n_splits=10, params=None, base_year=2019, preset='primary'):
    """Fixed-parameter out-of-fold predictions; no PSO or nested CV claim."""
    if stage not in (1, 2):
        raise ValueError('stage must be 1 or 2')
    if stage == 1 and params is None:
        raise ValueError('Stage1 requires explicit params')
    f, _ = training_frame(frame, stage, base_year)
    parts = []
    for fold, (tr, te) in enumerate(validation_splits(f, kind, n_splits)):
        fitted = (fit_stage1(f.iloc[tr], params=params, base_year=base_year) if stage == 1 else
                  fit_stage2(f.iloc[tr], params=params, base_year=base_year, preset=preset))
        out = fitted.predict(f.iloc[te])
        out['fold'] = fold
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def metrics(observed, predicted):
    a, b = np.asarray(observed, float), np.asarray(predicted, float)
    if len(a) < 2 or a.shape != b.shape or not np.isfinite([a, b]).all():
        raise ValueError('Metrics require >=2 aligned finite pairs; no implicit filtering')
    r = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None
    return dict(n=len(a), R2=float(r2_score(a, b)), R=r,
                RMSE=float(np.sqrt(mean_squared_error(a, b))), MAE=float(mean_absolute_error(a, b)))


def temporal_holdout(frame, *, preset='primary'):
    """Stage2 train 2019–2022, test all eligible 2023 rows; no residual cuts."""
    f = prepare(frame)
    train = f.loc[f.time.dt.year.between(2019, 2022)]
    test = f.loc[f.time.dt.year.eq(2023)].copy()
    if test.empty:
        raise ValueError('No 2023 holdout rows')
    model = fit_stage2(train, preset=preset)
    out = model.predict(test)
    out['stage2_preset'] = preset
    if 'XCH4_go' not in out:
        raise ValueError('Missing XCH4_go for holdout evaluation')
    out['evaluation_eligible'] = np.isfinite(out[['XCH4_go', 'XCH4_pred']]).all(axis=1)
    valid = out.loc[out.evaluation_eligible]
    scores = {name: metrics(valid.XCH4_go, valid[col]) for name, col in
              [('baseline', 'XCH4_pred'), ('corrected', 'XCH4_bc')]}
    return model, out, scores


def export_netcdf(frame, path):
    """Sparse records, not a Cartesian national grid; preserves input record order."""
    f = prepare(frame)
    cols = ['XCH4_go', 'XCH4_pred', 'predicted_delta', 'XCH4_bc']
    if any(c not in f for c in cols):
        raise ValueError(f'Export requires {cols}')
    ds = xr.Dataset({c: ('sample', f[c].to_numpy()) for c in cols}, coords={
        'sample': np.arange(len(f)), 'row_id': ('sample', f.row_id.to_numpy()),
        'time': ('sample', f.time.to_numpy()), 'lat': ('sample', f.lat.to_numpy()),
        'lon': ('sample', f.lon.to_numpy())})
    for c in cols:
        ds[c].attrs['units'] = 'ppb'
    ds.XCH4_go.attrs['long_name'] = 'Collocated GOSAT XCH4'
    ds.XCH4_pred.attrs['long_name'] = 'Archived first-stage Mapping XCH4, not TROPOMI observations'
    ds.XCH4_bc.attrs['long_name'] = 'Stage2 holdout bias-corrected XCH4'
    ds.predicted_delta.attrs['long_name'] = 'Predicted GOSAT minus Mapping residual'
    ds.lat.attrs['units'] = 'degrees_north'
    ds.lon.attrs['units'] = 'degrees_east'
    ds.time.attrs['long_name'] = 'Month of source collocation record'
    ds.row_id.attrs['long_name'] = 'Source CSV zero-based row identifier unless supplied explicitly'
    ds.attrs.update(title='Stage2 sample collocation predictions', base_year=2019,
                    coverage='Sparse sample records; not the complete China monthly 0.1 degree product',
                    training_period='2019-2022', holdout_period='2023')
    if 'stage2_preset' in f:
        ds.attrs['stage2_preset'] = str(f.stage2_preset.iloc[0])
    ds.to_netcdf(path, engine='netcdf4')
    ds.close()


def run_example(data_path='data/stage2_collocations_sample.csv', output_dir='outputs', *, preset='archived_loyo'):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    model, predictions, scores = temporal_holdout(pd.read_csv(data_path), preset=preset)
    predictions.to_csv(outdir / 'holdout_predictions.csv', index=False)
    export_netcdf(predictions, outdir / 'holdout_predictions.nc')
    model.save(outdir / 'stage2_model.joblib')
    (outdir / 'metrics.json').write_text(json.dumps(scores, indent=2) + '\n')
    return scores


def plot_saved(csv_path='outputs/holdout_predictions.csv', output_path='outputs/holdout_scatter.png'):
    """Plot only saved predictions; never fit models here."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    f = pd.read_csv(csv_path)
    f = f.loc[f.evaluation_eligible.astype(str).str.lower().eq('true')]
    fig = Figure(figsize=(9, 4))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    low = f[['XCH4_go','XCH4_pred','XCH4_bc']].min().min() - 5
    high = f[['XCH4_go','XCH4_pred','XCH4_bc']].max().max() + 5
    for ax, col, title in zip(axes, ['XCH4_pred','XCH4_bc'], ['Archived Mapping baseline','Stage2 correction']):
        ax.scatter(f.XCH4_go, f[col], s=15, alpha=.65)
        ax.plot([low,high], [low,high], color='black', linestyle='--', linewidth=1)
        ax.set(xlabel='Collocated GOSAT XCH4 (ppb)', title=title, xlim=(low,high), ylim=(low,high))
    axes[0].set_ylabel('Predicted XCH4 (ppb)')
    preset = f.stage2_preset.iloc[0] if 'stage2_preset' in f else 'primary'
    fig.suptitle(f'2023 sample holdout (n={len(f)}), {preset}\nStage2 trained on 2019–2022')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    return fig


if __name__ == '__main__':
    print(json.dumps(run_example(), indent=2))
