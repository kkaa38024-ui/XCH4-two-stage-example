# Input table

Provide a prepared monthly collocation CSV. The observational dataset is not included in this code-only distribution.

Required columns for Stage2:

| Column | Meaning |
|---|---|
| `time` | Calendar month as a date |
| `lat`, `lon` | Coordinates in degrees north/east |
| `t2m,tp,v10,u10,ssr,sp` | Prepared meteorological predictors |
| `NDVI,LST,DEM` | Prepared ancillary predictors |
| `XCH4_go` | Collocated GOSAT XCH4, ppb |
| `XCH4_pred` | Previously calculated Mapping XCH4, ppb |

Use the ancillary variables' prepared-study scales consistently; this code does not infer or convert their physical units. Leave missing numeric predictors blank. Dates and observation targets must be genuine. `t_index` and seasonal features are calculated by the code.

The default file name is `data/stage2_collocations_sample.csv`. A different file can be used from Python:

```python
from two_stage import run_example
run_example(data_path="path/to/prepared_collocations.csv", output_dir="outputs")
```

Stage1 additionally requires observed TROPOMI `XCH4_tr`; Mapping predictions or GOSAT values must not be relabelled as TROPOMI observations.
