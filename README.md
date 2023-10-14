# HEAL
## Requirements
The environment is provided in the conda format (`environment.yaml`). 
Please run `conda env create -f environment.yaml` to create the required environment.

## Root Cause Analysis
The root cause analysis algorithm is in `ours_full_pipeline.py`. 

We provide a test case data in `sample_data/anomaly_host_metrics.csv`. 
Run `python ours_full_pipeline.py` to get the test output on this case.

## Others (Visualization, history evolution diagram, etc.)
These visualization and analyisis code is in `visualization-analysis.ipynb`.

## Datasets
Due to commercial secrecy requirement, the host anomaly data is not provided.
We only provide the sample data of 1 anomaly case.

## Experiments
The cross-evaluation experiment code and the baselines are concluded in `exp-analysis.ipynb`.
For simplicity, we do not provide the intermediate results and datasets. This code is only for the demonstration purpose.