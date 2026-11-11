# Final V6 training package

Final source: FOUR_MODEL_TF215_DUALGPU_V6_final. Four models share frozen site folds and input identity checks. Use `run.py --help`; default execution is plan-only. TensorFlow 2.15.1 / Keras 2; dual-GPU synchronized batch normalization for neural models. RF runs on CPU.

`reference/original_code` supports AST consistency checks; it is not a second recommended training entry point. The release regenerates checksums after replacing private path defaults. Existing research result identities are not relabeled as this release.

Tests requiring TensorFlow or full research inputs have not all been executed during packaging. See ../docs/REPRODUCIBILITY.md.
