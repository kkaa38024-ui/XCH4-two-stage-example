# Two-stage XCH4 modelling: example code

Illustrative Python code for the two-stage TROPOMI/GOSAT XCH4 workflow associated with a China monthly 2019–2023 study.

1. **Stage1 — Mapping:** environmental predictors and time are used to reconstruct TROPOMI XCH4. An optional PSO helper selects XGBoost parameters on an internal development split.
2. **Stage2 — bias correction:** XGBoost learns the residual between collocated GOSAT XCH4 and Mapping predictions, then adds the predicted correction to Mapping.

This repository demonstrates the core methods and example usage; it is not a complete reproduction package for every analysis in the manuscript.

## Quick start

Python 3.11 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

For the observational example, prepare the input table described in `DATA_README.md`, place it at `data/stage2_collocations_sample.csv`, then run:

```bash
python two_stage.py
python -c "from two_stage import plot_saved; plot_saved()"
```

Alternatively, open `example.ipynb` and run its cells in order. The Notebook is distributed without execution outputs. The repository contains code and an input specification; observational data are supplied separately. The tests use explicitly synthetic fixtures and do not require study data.

The example saves `outputs/holdout_predictions.csv`, `outputs/holdout_predictions.nc`, `outputs/metrics.json` and a local model file. The plotting command reads the saved predictions. NetCDF stores collocations, not a full national grid.

On macOS, XGBoost needs the OpenMP runtime. Install `libomp` through your Python/conda environment or the normal platform package manager if it is not already available. Windows users should select their usual scientific Python kernel.

## Parameters and inputs

The executable example uses `archived_loyo`: 13 predictors including seasonal sine/cosine, 800 trees and depth8. `primary` retains the earlier 11-predictor/400-tree example. These are explicit preset choices, not model selection based on the demonstration holdout. See `METHODS.md` for feature lists and usage.

For Stage1, supply a prepared table with genuine observed `XCH4_tr`. The Stage2 input's `XCH4_pred` is a Mapping prediction and is not a substitute for that target.

## 中文说明

本仓库是两阶段方法的示例代码：第一阶段重建TROPOMI，第二阶段用GOSAT作残差校正。先安装依赖；基础测试可直接运行。若要运行观测数据例程，按字段说明准备CSV，再打开Notebook或执行主程序。这里只演示核心流程，不包含论文全部实验。
