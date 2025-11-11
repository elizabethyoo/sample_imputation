# Sample Imputation

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

determine missingness paterns in sample beiwe data, and try imputation methods

## MIMS Processing

This repository now includes a Python port of the Monitor Independent Movement Summary (MIMS) algorithm proposed by Zhang et&nbsp;al. in the context of raw accelerometer processing [[Zhang et&nbsp;al., 2012]](https://doi.org/10.1123/jmpb.2018-0068). The implementation mirrors the reference R package (mHealthGroup/MIMSunit) and lives under `src/mims/mims_unit.py`.

```python
import pandas as pd
from mims import mims_unit

df = pd.read_csv("my_accelerometer.csv", parse_dates=["timestamp"])
summary = mims_unit(
    df,
    epoch="1 sec",
    dynamic_range=(-8.0, 8.0),
    output_mims_per_axis=True,
)
```

Guard rails:

- The first column must be a monotonic timestamp without duplicates. Remaining columns must be acceleration axes in `g`.
- The `dynamic_range` tuple should match the sensor range; values outside the range are clipped before integration.
- Set `use_filtering=False` to skip the 0.2–5 Hz Butterworth band-pass cascade during experimentation.

See the module docstring in `src/mims/mims_unit.py` for additional configuration parameters and implementation notes.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Metadata from Beiwe Service Platform, etc.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│       ├──  
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks
│   ├── 00_plot_hr_acc_data.ipynb
│   └── 01_summarize_hr_acc.ipynb        
├── pyproject.toml     <- Project configuration file 
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
├── setup.cfg          <- Configuration file for flake8
│
└── sample_imputation   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes sample_imputation a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

