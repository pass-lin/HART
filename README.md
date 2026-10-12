# HART: Hybrid Attention-RWKV Time series Foundation Model

**Paper:** [HART: Hybrid Attention-RWKV Time series Foundation Model](https://arxiv.org/abs/2610.12883)

**中文文档:** [README_zh.md](README_zh.md)

HART is a time series foundation model (about 36M parameters) that interleaves RWKV-7
blocks with causal Transformer blocks and outputs quantile forecasts. The released
checkpoint `Hart-36M` supports zero-shot evaluation on GIFT-Eval and the TIME benchmark.

## Directory layout

```
opensorce/
├── gifteval.py                 # Zero-shot evaluation on GIFT-Eval
├── timebench_eval.py           # Zero-shot evaluation on the TIME benchmark
├── dataset_properties.json     # GIFT-Eval dataset property mapping
├── model/RWKV_Time/            # Model definition
├── weights/Hart-36M/           # config.json and model.weights.h5
└── results/
    ├── gift_eval/              # GIFT-Eval results
    └── time_bench/Hart-36M/    # TIME benchmark results
```

## Model configuration

The parameters in `weights/Hart-36M/config.json` are the default architecture values.

| Parameter | Value |
| --- | --- |
| hidden_size | 512 |
| head_size | 64 |
| num_layers | 6 |
| intermediate_dim | 1024 |
| n_quantiles | 9 |
| patch_size | 16 |
| mix_transformer_layer_num | 6 |

## Evaluation

```bash
python gifteval.py --model_path weights/Hart-36M --cuda_devices 0 --max_length 2048
python timebench_eval.py --model_path weights/Hart-36M --cuda_devices 0 --max_length 2048
```

- `--cuda_devices`: visible GPU index.
- `--model_path`: model directory containing `config.json` and `model.weights.h5`.
- `--max_length`: upper bound on the number of patches; multiplied by `patch_size` to
  obtain the actual token length.
- The GIFT-Eval data root is read from the `GIFT_EVAL` environment variable
  (default: `gift_eval`).
- The TIME benchmark data root is read from the `TIME_DATASET` environment variable
  (default: `TIME/data`).

## Results

- GIFT-Eval: `full_results.csv`, `average_results.csv` and `domain_average_results.csv`
  under `results/gift_eval/`.
- TIME benchmark: per-task `predictions.npz`, `metrics.npz` and `config.json` under
  `results/time_bench/Hart-36M/<dataset>/<freq>/<term>/`, aggregated into
  `full_results.csv` and `average_results.csv` at the model directory level.

## Citation

If you find HART useful, please cite the paper:

```bibtex
@article{hart2026hybrid,
  title   = {HART: Hybrid Attention-RWKV Time series Foundation Model},
  journal = {arXiv preprint arXiv:2610.12883},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.12883}
}
```
# HART
