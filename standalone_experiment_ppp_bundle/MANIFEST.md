# Standalone Experiment PPP Bundle

This folder contains the notebooks, Excel exports, generated plots, and local Python source files needed to inspect and rerun the PPP plotting workflow outside the original repository.

## Notebooks
- `notebooks/expermint_ppp.ipynb`
- `notebooks/graphs_historical.ipynb`
- `notebooks/graphs_synthetic.ipynb`

## Key Plot Outputs
- `plots/experiment_ppp_assets/plot_01_cell11_out2.png`
- `plots/experiment_ppp_assets/plot_02_cell13_out1.png`
- `plots/historical/figure_18_extended_backtest_oos_r2_vs_rff.png`
- `plots/historical/figure_19_extended_backtest_spectral_diagnostics.png`
- `plots/historical/figure_20_extended_backtest_subspace_stability_diagnostics.png`
- `plots/historical/historical_feature_diagnostics.png`
- `plots/historical/historical_oos_r2_vs_rff.png`
- `plots/historical/historical_portfolio_performance.png`
- `plots/historical/portfolio_performance.png`
- `plots/synthetic/figure_08_synthetic_t24_fitted_rank_comparison.png`
- `plots/synthetic/figure_09_synthetic_t24_stability_summaries.png`
- `plots/synthetic/figure_10_synthetic_rank_aligned_window_length_comparison.png`
- `plots/synthetic/figure_11_synthetic_rank_aligned_stability_window_comparison.png`
- `plots/synthetic/synth_v5_main_compare.png`
- `plots/synthetic/synth_v5_stability_compare.png`
- `plots/synthetic/synth_v5_window_compare.png`
- `plots/synthetic/synth_v5_window_stability.png`

## Root Excel Exports
- `features_data_12_24.xlsx`
- `features_data_14_36.xlsx`
- `features_data_16_24.xlsx`
- `features_data_16_36.xlsx`
- `features_data_18_24.xlsx`
- `features_data_18_36.xlsx`
- `features_data_all.xlsx`
- `features_data_all_14.xlsx`
- `features_data_all_factors.xlsx`
- `features_data_all_factors_10.xlsx`
- `features_data_all_factors_12_seeds.xlsx`
- `features_data_all_factors_12_seeds_10.xlsx`
- `features_data_all_factors_14_seeds.xlsx`
- `features_data_synthetic_17_24.xlsx`
- `features_data_synthetic_18_24.xlsx`
- `features_data_synthetic_18_36.xlsx`
- `oos_r2_12_24.xlsx`
- `oos_r2_14_24.xlsx`
- `oos_r2_14_36.xlsx`
- `oos_r2_18_24.xlsx`
- `oos_r2_18_36.xlsx`
- `oos_r2_synthetic_17_24.xlsx`
- `oos_r2_synthetic_18_24.xlsx`
- `oos_r2_synthetic_18_36.xlsx`
- `portfolio_performance_12_24.xlsx`
- `portfolio_performance_14_36.xlsx`
- `portfolio_performance_16_24.xlsx`
- `portfolio_performance_16_36.xlsx`
- `portfolio_performance_18_24.xlsx`
- `portfolio_performance_18_36.xlsx`
- `portfolio_performance_alpha.xlsx`
- `portfolio_performance_alpha_12.xlsx`
- `portfolio_performance_alpha_14.xlsx`
- `portfolio_performance_alpha_factors.xlsx`
- `portfolio_performance_alpha_factors_10.xlsx`
- `portfolio_performance_factors_seeds_12.xlsx`
- `portfolio_performance_factors_seeds_14.xlsx`
- `portfolio_performance_synthetic_17_24.xlsx`
- `portfolio_performance_synthetic_18_24.xlsx`
- `portfolio_performance_synthetic_18_36.xlsx`
- `prediction_stats_all.xlsx`

## Data Excel Files
- `data/definitions.xlsx`
- `data/openap_subset_10f_20s_10y.xlsx`

## Included Runtime Files
- `src/`
- `ipca/`
- `data/`
- `requirements.txt` if present in the source repository

## Notes
- The bundled `notebooks/expermint_ppp.ipynb` was adjusted to set its working directory to this bundle root instead of the original absolute path.
- Large unrelated pickle caches named `my_data*.pkl` / `your_data.pkl` were not included. The current notebook looks for `result_rff_all_*.pkl` only if `LOAD_RESULTS_FROM_PICKLES=True`; no such pickle files were present in the source repository.
- Historical and synthetic graph notebooks read the Excel exports from this folder root and save figures under `plots/historical/` and `plots/synthetic/`.
- The historical and synthetic graph notebooks were test-run from this folder after packaging. The first import cell of `expermint_ppp.ipynb` was also checked from this folder.
