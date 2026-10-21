# Chaotic Dynamics Forecasting

Research code and reports by Zhang YUE for forecasting the Kuramoto–Sivashinsky (KS), Lorenz–96, and two-dimensional Kolmogorov systems. The experiments use self-attention, Easy Attention with a feature gate, and LSTM.

## Reports

- [Short research report](reports/Zhang_YUE_Project_Report.pdf): the forecasting methods, the development of the KS pipeline, and the main NRMSE and valid prediction time comparisons.
- [Detailed KS report](reports/Zhang_YUE_Complete_Report.pdf): the KS system, model formulations, implementation, experiments, and extended discussion.

## Code

| Directory | Contents |
| --- | --- |
| `code/9.18/downloaded_results/` | Source and configurations saved with the final KS runs at `ds = 16` and `ds = 32`, plus the three-model comparison script. |
| `code/KS_3Models_Block100_fixed/` | KS implementation for 100-step block forecasts. |
| `code/Lorenz 96/` | Lorenz–96 implementations and configurations. |
| `code/2D_Kolmogorov/` | Two-dimensional Kolmogorov implementations and configurations. |
| `code/LSTM_5DS_Custom(失败案例）/` | Earlier LSTM experiments across five spatial sampling factors. |

The main KS comparison observes four spatial channels at `ds = 16` and two at `ds = 32`. Each run retains its own source snapshot and configuration. The reports describe the evaluation protocol and results.

The Python code uses PyTorch, NumPy, h5py, Matplotlib, and PyYAML. Set the data and launcher paths for your environment before running an experiment.
