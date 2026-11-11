# HLS footprint carbon flux

Code release for footprint-aware estimation of NEE, GPP and RECO using HLS imagery and half-hourly environmental data.

## Organization / 目录

- `training/`: final V6 four-model training package: FAT, uniform-weighted Transformer, preaggregation Transformer and RF. Internal names are retained to preserve imports and verification.
- `analysis/`: final FAT-V6 paper analysis scripts, grouped by accuracy, robustness, gain correlates, RF importance and spatial case studies.
- `examples/`: synthetic-data generator and the 110-column flux/environment schema. No measured site records are included in the examples.
- `docs/`: data layout, reproducibility notes, portable path defaults and release checksums.

## Quick start / 示例

Run from the repository root (Python 3.10 recommended):

```bash
python -m pip install -r requirements.txt
python examples/generate_example.py
python examples/validate_example.py
python training/verify.py
python training/test_batching.py
```

The generated arrays illustrate interfaces only. They do not reproduce paper results and must not be substituted for the frozen research dataset in training audits.

## Training / 训练

```bash
python -m pip install -r requirements-training.txt
python training/run.py --model FAT --fold 0 --input-root ./data/input_data
```

This prints a plan. Add `--execute` only with the real inputs matching the frozen input manifest and the intended GPU environment. V6 targets TensorFlow 2.15.1/Keras 2 and two GPUs; this release does not claim a single-GPU reproduction. Existing fold outputs are protected against accidental overwrite. See `docs/DATA_FORMAT.md` and `docs/REPRODUCIBILITY.md`.

本仓库按最终版整理，个人机器路径改为相对路径。原始观测、大型影像、模型权重、账号凭据和运行日志不随代码公开。模型结构、损失与评价公式保持原实现；训练、论文分析的环境与指标符号差异见复现说明。

No license is granted beyond the rights provided by the hosting platform; a research software license has not yet been selected by the author.
