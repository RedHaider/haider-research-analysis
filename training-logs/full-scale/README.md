# Full training portfolio

64 completed experiments for 32 professor pairs. Each row links the two studies, their held-out metrics, and the shared training code. These are independent research benchmarks. Several companions apply simulated perturbations to measured inputs.

See [held-out comparisons for the first group](RESULTS.md) and [held-out comparisons for the final 15 pairs](remaining/RESULTS.md).

| Professor | Research topic | Professor-aligned study | Public-data companion | Trainer |
| --- | --- | --- | --- | --- |
| Saeed Amal | Weak-label pathology bags | [Study](../../professor-aligned/amal-pathology-mil-aligned/) | [Study](../../real-data-projects/amal-pathology-mil-real-data/) | [Code](pathology/train.py) |
| Polina Golland | Loss-conditioned segmentation | [Study](../../professor-aligned/golland-loss-conditioned-segmentation/) | [Study](../../real-data-projects/golland-busbra-contour-probability/) | [Code](segmentation/train.py) |
| Jeff Leek | Frozen nuisance correction | [Study](../../professor-aligned/leek-frozen-surrogate-shift/) | [Study](../../real-data-projects/leek-metabric-frozen-correction/) | [Code](surrogate/train.py) |
| James Zou | Out-of-bag data valuation | [Study](../../professor-aligned/zou-out-of-bag-data-valuation/) | [Study](../../real-data-projects/zou-covertype-data-valuation/) | [Code](valuation/train.py) |
| René Vidal | Sparse subspace clustering | [Study](../../professor-aligned/vidal-sparse-subspace-recovery/) | [Study](../../real-data-projects/vidal-fashion-anchor-subspaces/) | [Code](subspaces/train.py) |
| Qingyun Wu | Cost-aware AutoML | [Study](../../professor-aligned/wu-fidelity-automl/) | [Study](../../real-data-projects/wu-electricity-frugal-search/) | [Code](automl/train.py) |
| Guy Brock | Expression cluster stability | [Study](../../professor-aligned/brock-expression-cluster-validation/) | [Study](../../real-data-projects/brock-pbmc-cluster-stability/) | [Code](stability/train.py) |
| Suhang Wang | Label-wise graph learning | [Study](../../professor-aligned/wang-label-wise-graph/) | [Study](../../real-data-projects/wang-amazon-heterophilic-graph/) | [Code](graphs/train.py) |
| Qunhua Li | Replicate reproducibility | [Study](../../professor-aligned/qunhua-li-replicate-mixture/) | [Study](../../real-data-projects/qunhua-li-encode-reproducibility/) | [Code](reproducibility/train.py) |
| Walter Scheirer | Open-set recognition | [Study](../../professor-aligned/scheirer-extreme-tail-recognition/) | [Study](../../real-data-projects/scheirer-cifar-open-set/) | [Code](open-set/train.py) |
| Paul Bogdan | Fractional-memory signal forecasting | [Study](../../professor-aligned/bogdan-eeg-fractional-forecast/) | [Study](../../real-data-projects/bogdan-wisdm-fractional-forecast/) | [Code](fractional/train.py) |
| Mehak Gupta | Flexible ICU prediction windows | [Study](../../professor-aligned/mehak-consistent-window-forecast/) | [Study](../../real-data-projects/mehak-icu-flexible-horizons/) | [Code](flexible-windows/train.py) |
| Jihun Hamm | Test-time adaptation | [Study](../../professor-aligned/hamm-frozen-output-adaptation/) | [Study](../../real-data-projects/hamm-sensor-stream-adaptation/) | [Code](stream-adaptation/train.py) |
| Guha Balakrishnan | Volumetric deformable registration | [Study](../../professor-aligned/balakrishnan-deformation-integration/) | [Study](../../real-data-projects/balakrishnan-hippocampus-registration/) | [Code](registration/train.py) |
| Alex K. Jones | Learned cache admission | [Study](../../professor-aligned/jones-two-tier-memory-traffic/) | [Study](../../real-data-projects/jones-twitter-cache-admission/) | [Code](cache-replay/train.py) |
| Huixin Zhan | Multi-objective battery dispatch | [Study](../../professor-aligned/zhan-multiobjective-dispatch/) | [Study](../../real-data-projects/zhan-battery-measured-load-policy/) | [Code](multiobjective/train.py) |
| Ruyi Lian | Coarse-to-fine CAD correspondences | [Study](../../professor-aligned/lian-progressive-pose-codes/) | [Study](../../real-data-projects/lian-tudl-cad-correspondences/) | [Code](pose/train.py) |
| Jia Li | Spatial image annotation | [Study](../../professor-aligned/li-spatial-annotation/) | [Study](../../real-data-projects/li-cifar-semantic-regions/) | [Code](remaining/spatial_tags.py) |
| Ehsan Azimi | Anisotropic overlay calibration | [Study](../../professor-aligned/azimi-anisotropic-hmd-calibration/) | [Study](../../real-data-projects/azimi-tudl-overlay-calibration/) | [Code](remaining/overlay.py) |
| Aggelos Katsaggelos | Learned reconstruction prior | [Study](../../professor-aligned/katsaggelos-plugplay-phantoms/) | [Study](../../real-data-projects/katsaggelos-mri-consistency/) | [Code](remaining/plugplay.py) |
| Jian Peng | Multiplex diffusion | [Study](../../professor-aligned/peng-multiplex-diffusion/) | [Study](../../real-data-projects/peng-human-network-integration/) | [Code](remaining/network_diffusion.py) |
| Juan Cui | Cross-omics recovery | [Study](../../professor-aligned/cui-mirna-integrative-network/) | [Study](../../real-data-projects/cui-paad-cross-omics/) | [Code](remaining/cross_omics.py) |
| Liqing Zhang | Semi-supervised assay adaptation | [Study](../../professor-aligned/zhang-semi-supervised-assay-transfer/) | [Study](../../real-data-projects/zhang-tcga-assay-shift/) | [Code](remaining/assay_transfer.py) |
| Yuri Pritykin | Tumor immune state analysis | [Study](../../professor-aligned/pritykin-cell-mixture-states/) | [Study](../../real-data-projects/pritykin-melanoma-cell-states/) | [Code](remaining/tumor_states.py) |
| Linwei Wang | Missing ECG lead morphology | [Study](../../professor-aligned/wang-latent-inverse-leads/) | [Study](../../real-data-projects/wang-ptbxl-lead-morphology/) | [Code](remaining/lead_recovery.py) |
| Daniel Moyer | Scanner-invariant learning | [Study](../../professor-aligned/moyer-scanner-invariant-encoding/) | [Study](../../real-data-projects/moyer-busbra-scanner-shift/) | [Code](remaining/scanner_invariance.py) |
| Ajita Rattani | EMG time-frequency classification | [Study](../../professor-aligned/rattani-emg-time-frequency/) | [Study](../../real-data-projects/rattani-uci-emg-gestures/) | [Code](remaining/gesture.py) |
| Jun Bai | Tissue region graphs | [Study](../../professor-aligned/bai-weak-label-region-graph/) | [Study](../../real-data-projects/bai-pathmnist-region-graph/) | [Code](remaining/tissue_graph.py) |
| Amir A. Amini | PSF-aware motion recovery | [Study](../../professor-aligned/amini-psf-motion-estimation/) | [Study](../../real-data-projects/amini-cardiac-mri-motion/) | [Code](remaining/motion_field.py) |
| Xiaobai Liu | Contextual trajectory decoding | [Study](../../professor-aligned/liu-trajectory-context-graph/) | [Study](../../real-data-projects/liu-har-contextual-decoding/) | [Code](remaining/trajectory.py) |
| David Dickensheets | Microscope focus classification | [Study](../../professor-aligned/dickensheets-optical-section-confidence/) | [Study](../../real-data-projects/dickensheets-bbbc-microscope-focus/) | [Code](remaining/focus.py) |
| Sandeep Singhal | Clinical and expression fusion | [Study](../../professor-aligned/singhal-multimodal-risk-fusion/) | [Study](../../real-data-projects/singhal-metabric-risk-fusion/) | [Code](remaining/oncology_fusion.py) |

Trainers use a local datasets directory. The training notes give dataset sources and run commands. This repository contains code and aggregate metrics.
