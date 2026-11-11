# Reproducibility scope

This release packages the final V6 training source and final FAT-V6 analysis source. It is not a claim that all four historical paper predictions were trained in one identical environment: the final analysis README records V6 FAT/Uniform and earlier local Preaggregation/RF results. Regenerating a fully controlled V6 comparison requires rerunning all four models in the specified environment.

Training evaluation uses its original bias convention; paper Table S4 scripts define Bias = prediction minus observation. Consult each script before comparing signs. FAT versus Uniform isolates the aggregation rule; FAT versus preaggregation changes additional processing and is a method comparison, not a strict one-factor ablation.

The required frozen splits (not distributed) and released input hashes apply to the original research dataset, not arbitrary new data. Synthetic examples validate shape/alignment and footprint normalization only. Full training, real-data figure reproduction, GPU numerical equivalence and case-study inference are not verified by these example tests.

Case-study pixel inference currently calls the legacy streaming inference code under `data/external/HLS_ERA5_FLUX_FFP_CV5_FINAL`; that external dependency and trained weights must be supplied. It is not part of the four-model training entry point. Ancillary land-cover/driver tables and map resources are also external inputs. The publication's raw-data redistribution terms must be handled separately.

Personal absolute paths were replaced with portable relative defaults. Run analysis commands from the repository root. Code locks and sanitized reference hashes are regenerated for this release; they must not be presented as the original training-run hashes. `release_manifest.json` retains original source hashes without personal paths.
