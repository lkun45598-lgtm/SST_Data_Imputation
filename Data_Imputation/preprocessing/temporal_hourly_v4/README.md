# Causal Hourly JAXA Proxy V4

This isolated pipeline rebuilds the nine JAXA SST series in real timestamp
order. It treats only `original_obs_mask == 1` finite ocean pixels as raw
observations, uses up to 72 prior real hours for same-location temporal fill,
and uses confidence-limited historical synthetic structure only for residual
spatial reconstruction.

The source root is the existing hourly layout:

```text
Data_Imputation/experiments/hourly_data/h00 ... h23
```

Outputs use the same direct `h00`--`h23` H5 layout. Core legacy fields retain
their dtype and compression. V4 additionally writes `source_code`,
`proxy_confidence`, `knn_fill_mask`, and `unresolved_mask` for auditability.
The output root must be empty. The command refuses existing content rather
than overwriting it.

Run a one-series causal smoke job through a chosen timestamp:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m Data_Imputation.preprocessing.temporal_hourly_v4.cli \
  --source-root Data_Imputation/experiments/hourly_data \
  --output-root /data/sst_data/sst_missing_value_imputation/jaxa_processed_data_smoke_v4 \
  --series 8 --smoke-until 2024-07-30T23:00:00 --query-workers 21
```

After smoke figures and diagnostics have been reviewed, regenerate the full
approved output with all nine independent series. A series stays chronological;
only different series run concurrently.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m Data_Imputation.preprocessing.temporal_hourly_v4.cli \
  --source-root Data_Imputation/experiments/hourly_data \
  --output-root /data/sst_data/sst_missing_value_imputation/jaxa_processed_data \
  --all-series --full --series-workers 9 --query-workers 21
```

Each run records per-frame CSV/JSON diagnostics and a root manifest under
`diagnostics/`. Low-confidence synthetic KNN values remain in the frame that
needs them, but are removed from the next frames' proxy history.
