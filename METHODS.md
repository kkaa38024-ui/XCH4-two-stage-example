# Example methods

## Stage1: Mapping

Predictors, in order: `t2m,tp,v10,u10,ssr,sp,NDVI,LST,DEM,t_index`.
The target is observed TROPOMI `XCH4_tr`. Calendar time is anchored to January2019 and is retained when predicting later subsets.

`fit_stage1` fits a model using an explicit parameter dictionary. Optional `tune_stage1` searches the original primary PSO ranges: trees200–800, depth4–10, learning rate0.01–0.2, subsample and column sample0.6–1.0. Its internal development split is60/40; the default budget is16particles and8iterations. This helper is not nested cross-validation.

## Stage2: correction

Learn `delta = XCH4_go - XCH4_pred`; return `XCH4_bc = XCH4_pred + predicted_delta`.

- `primary`: Mapping plus the ten Stage1 predictors; 400trees, depth6, learning rate0.05, subsample0.8, column sample0.8.
- `archived_loyo`: the same eleven predictors plus `month_sin,month_cos`; 800trees, depth8, learning rate0.02, subsample0.8, column sample0.9. Seasonal features are sine/cosine of2π×calendar month/12. This preset follows the archived validation model configuration; the example does not itself rerun the whole archived five-year study.

Both presets use squared-error histogram trees, seed42 and two CPU threads. The API retains `primary` as its default; the executable example explicitly selects `archived_loyo`.

## Example evaluation and outputs

The observational example fits Stage2 on2019–2022 rows and predicts2023. Baseline and corrected metrics use the same eligible rows. Missing predictors stay NaN; missing observation labels are not imputed. Scaling is fitted on training rows only, row identities remain aligned, and no error-based sample filtering is performed.

`cross_validate` supports sample, monthly and spatial groups when corresponding inputs are provided. Model saving retains preprocessing, feature order and time anchoring. Prediction CSV and sparse collocation NetCDF use ppb for XCH4. Figures read saved predictions rather than refitting the model.
