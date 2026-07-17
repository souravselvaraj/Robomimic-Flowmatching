# Reproduction configs

Training configs for the benchmark results in the top-level README.

- `benchmark/{fm,dp}_{lift,can,square,tool_hang,transport}_seed{1,2,3}.json` —
  the 30-run, 3-seed sweep ({flow_matching, diffusion_policy} × 5 tasks × 3
  seeds), matched budget (1000 epochs × 100 steps, batch 256). Regenerate with
  [`scripts/gen_benchmark_configs.py`](../scripts/gen_benchmark_configs.py).
- `fm_square_lowdim.json` — single-task flow-matching config for NutAssemblySquare.

**Dataset / output paths:** these configs record the exact absolute paths used on
the cluster where the results were produced (`.../robomimic_data/<task>/ph/low_dim_v15.hdf5`
and an `output/` dir). Edit `train.data` and `train.output_dir` for your machine,
or regenerate with the script after downloading the datasets the standard way:

```bash
python robomimic/scripts/download_datasets.py \
    --tasks lift can square tool_hang transport \
    --dataset_types ph --hdf5_types low_dim
```

Run one config through the full robomimic pipeline:

```bash
python scripts/train.py --config configs/benchmark/fm_square_seed1.json
# scripts/train.py imports robomimic_cfm (registering "flow_matching") and then
# hands off to robomimic's own train.py entry point.
```
