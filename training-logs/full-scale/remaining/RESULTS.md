# Held-out results for the 15 final professor pairs

The table reports each public-data companion. The two project directories per professor contain the complete metrics, fitted models and held-out data. See the [study index](README.md) for research references, dataset details and limitations.

| Professor | Metric | Baseline | Tested method | Aligned metrics | Public-data metrics |
| --- | --- | --- | --- | --- | --- |
| Jia Li | macro-F1 | Global descriptors 0.318 | Spatial descriptors 0.456 | [JSON](../../../professor-aligned/li-spatial-annotation/results.json) | [JSON](../../../real-data-projects/li-cifar-semantic-regions/results.json) |
| Ehsan Azimi | mean corner error (px) | Unweighted calibration 1.290 | Learned precision 1.011 | [JSON](../../../professor-aligned/azimi-anisotropic-hmd-calibration/results.json) | [JSON](../../../real-data-projects/azimi-tudl-overlay-calibration/results.json) |
| Aggelos Katsaggelos | mean PSNR (dB) | Median prior 18.157 | Learned prior 18.355 | [JSON](../../../professor-aligned/katsaggelos-plugplay-phantoms/results.json) | [JSON](../../../real-data-projects/katsaggelos-mri-consistency/results.json) |
| Jian Peng | macro average precision | Pooled diffusion 0.129 | Network-specific diffusion 0.121 | [JSON](../../../professor-aligned/peng-multiplex-diffusion/results.json) | [JSON](../../../real-data-projects/peng-human-network-integration/results.json) |
| Juan Cui | scaled RMSE | Ridge 0.7978 | Shared latent PLS 0.7962 | [JSON](../../../professor-aligned/cui-mirna-integrative-network/results.json) | [JSON](../../../real-data-projects/cui-paad-cross-omics/results.json) |
| Liqing Zhang | macro-F1 | Source only 1.000 | Semi-supervised adaptation 1.000 | [JSON](../../../professor-aligned/zhang-semi-supervised-assay-transfer/results.json) | [JSON](../../../real-data-projects/zhang-tcga-assay-shift/results.json) |
| Yuri Pritykin | patient-level AUROC | Mean pooling 0.967 | Distributional pooling 0.811 | [JSON](../../../professor-aligned/pritykin-cell-mixture-states/results.json) | [JSON](../../../real-data-projects/pritykin-melanoma-cell-states/results.json) |
| Linwei Wang | scaled RMSE | Ridge 0.708 | Nonlinear latent model 0.685 | [JSON](../../../professor-aligned/wang-latent-inverse-leads/results.json) | [JSON](../../../real-data-projects/wang-ptbxl-lead-morphology/results.json) |
| Daniel Moyer | AUROC | Task only 0.703 | Scanner-invariant training 0.706 | [JSON](../../../professor-aligned/moyer-scanner-invariant-encoding/results.json) | [JSON](../../../real-data-projects/moyer-busbra-scanner-shift/results.json) |
| Ajita Rattani | macro-F1 | Time only 0.831 | Time-frequency 0.838 | [JSON](../../../professor-aligned/rattani-emg-time-frequency/results.json) | [JSON](../../../real-data-projects/rattani-uci-emg-gestures/results.json) |
| Jun Bai | macro-F1 | Independent regions 0.776 | Region graph 0.814 | [JSON](../../../professor-aligned/bai-weak-label-region-graph/results.json) | [JSON](../../../real-data-projects/bai-pathmnist-region-graph/results.json) |
| Amir A. Amini | mean endpoint error (px) | Farneback 0.310 | PSF-conditioned model 0.333 | [JSON](../../../professor-aligned/amini-psf-motion-estimation/results.json) | [JSON](../../../real-data-projects/amini-cardiac-mri-motion/results.json) |
| Xiaobai Liu | macro-F1 | Local emission 0.940 | Contextual decoding 0.953 | [JSON](../../../professor-aligned/liu-trajectory-context-graph/results.json) | [JSON](../../../real-data-projects/liu-har-contextual-decoding/results.json) |
| David Dickensheets | AUROC | Single-channel focus 0.785 | Focus feature fusion 0.981 | [JSON](../../../professor-aligned/dickensheets-optical-section-confidence/results.json) | [JSON](../../../real-data-projects/dickensheets-bbbc-microscope-focus/results.json) |
| Sandeep Singhal | AUROC | Clinical only 0.674 | Clinical and expression 0.579 | [JSON](../../../professor-aligned/singhal-multimodal-risk-fusion/results.json) | [JSON](../../../real-data-projects/singhal-metabric-risk-fusion/results.json) |

Metrics are benchmark-specific and should be interpreted within their own dataset and split. Examples using simulated perturbations include the TUD-L overlay study, the MRI measurement masking and motion studies, and the RNA-seq assay shift.

The clinical fusion and small melanoma cohort studies are exploratory. The METABRIC study excludes records censored before five years. The melanoma model evaluates only 19 baseline patients, even though expression profiles were available for thousands of their cells.
