# Scaling Benchmark Results

## System Info

- Python version: `3.12.3`
- Platform: `Linux-6.8.0-100-generic-aarch64-with-glibc2.39`
- CPU cores: `6`

## Results

| Method | Wall Time (s) | Output Rows | Backend/Notes |
|---|---:|---:|---|
| chunking | 0.010616 | 300 | pandas |
| multiprocessing | 0.001483 | 300 | multiprocessing |
| dask | 0.015570 | 300 | dask |

## Artifacts

- Raw CSV: `/home/sp/cyber-course/projects/datascience/datasets/large_synthetic.csv`
- Chunked CSV: `/home/sp/cyber-course/projects/datascience/datasets/large_synthetic_processed.csv`
- Multiprocessing CSV: `/home/sp/cyber-course/projects/datascience/datasets/large_synthetic_processed_mp.csv`
- Dask/Parquet output: `/home/sp/cyber-course/projects/datascience/datasets/large_synthetic_processed.parquet`

Note: If Dask/Parquet dependencies are unavailable, the `.parquet` file may be an offline-safe placeholder artifact with benchmark metadata and preview rows.