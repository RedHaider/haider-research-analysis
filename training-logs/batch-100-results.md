# Batch 100 — larger neural studies

Forty trained studies for twenty professors. Both models use two seeds and validation-based model selection. Local CPU runs use up to eight epochs for the compact model and twelve for the larger model, with early stopping. The batch-size study uses the same large architecture and twelve-epoch budget in both arms.

| Professor | Study | Metric | Baseline arm | Comparison arm | Comparison parameters |
|---|---|---|---:|---:|---:|
| Xia Ning | [Professor-aligned simulation](../professor-aligned/clinical-risk-missing-labs-heavy-aligned/README.md) | auroc | 0.8222 | 0.8272 | 1,258,954 |
| Lijun Cheng | [Professor-aligned simulation](../professor-aligned/cardiac-feature-occlusion-heavy-aligned/README.md) | auroc | 0.7783 | 0.7769 | 681,730 |
| Lubomir Hadjiyski | [Professor-aligned simulation](../professor-aligned/mammography-analogue-radiomics-heavy-aligned/README.md) | auroc | 1.0000 | 1.0000 | 1,524,682 |
| Neil Abernethy | [Professor-aligned simulation](../professor-aligned/readmission-clinical-informatics-heavy-aligned/README.md) | auroc | 0.7781 | 0.7858 | 684,802 |
| Shuai Huang | [Professor-aligned simulation](../professor-aligned/short-horizon-glucose-risk-heavy-aligned/README.md) | auroc | 0.8984 | 0.9088 | 1,256,554 |
| Marc Niethammer | [Professor-aligned simulation](../professor-aligned/affine-registration-pairs-heavy-aligned/README.md) | localization_mae_px | 0.9088 | 0.6704 | 1,526,315 |
| Brent Munsell | [Professor-aligned simulation](../professor-aligned/morphometry-followup-heavy-aligned/README.md) | mae | 0.0175 | 0.0182 | 665,089 |
| Sayeed Ghani | [Professor-aligned simulation](../professor-aligned/inertial-wireless-loss-heavy-aligned/README.md) | auroc | 1.0000 | 0.9999 | 1,256,954 |
| Arghya Kusum Das | [Professor-aligned simulation](../professor-aligned/batched-clinical-learning-heavy-aligned/README.md) | elapsed_seconds | 2.5944 | 1.4115 | 673,282 |
| Yung-Hsiang Lu | [Professor-aligned simulation](../professor-aligned/edge-resolution-gating-heavy-aligned/README.md) | accuracy | 1.0000 | 1.0000 | 1,525,322 |
| Chaofan Chen | [Professor-aligned simulation](../professor-aligned/learned-image-prototypes-heavy-aligned/README.md) | auroc | 1.0000 | 1.0000 | 1,401,896 |
| Hong Xiao | [Professor-aligned simulation](../professor-aligned/temporal-signal-artifacts-heavy-aligned/README.md) | auroc | 1.0000 | 1.0000 | 1,257,354 |
| Dimitris N. Metaxas | [Professor-aligned simulation](../professor-aligned/masked-fourier-image-recovery-heavy-aligned/README.md) | psnr | 20.2290 | 24.3542 | 1,549,432 |
| Sharon Xiaolei Huang | [Professor-aligned simulation](../professor-aligned/polyp-segmentation-corruption-heavy-aligned/README.md) | dice | 0.7431 | 0.8592 | 1,550,072 |
| Xiaohu Guo | [Professor-aligned simulation](../professor-aligned/deformation-landmark-regression-heavy-aligned/README.md) | landmark_error_px | 1.0391 | 0.7968 | 1,528,080 |
| Suresh S. Muknahallipatna | [Professor-aligned simulation](../professor-aligned/vessel-segmentation-multiscale-heavy-aligned/README.md) | dice | 0.6206 | 0.8701 | 1,549,432 |
| Ngan Le | [Professor-aligned simulation](../professor-aligned/caption-noise-crossmodal-heavy-aligned/README.md) | auroc | 0.9995 | 0.9991 | 2,234,626 |
| Dongming Peng | [Professor-aligned simulation](../professor-aligned/vital-transition-forecast-heavy-aligned/README.md) | auroc | 0.4779 | 0.5472 | 1,257,354 |
| Changqing Li | [Professor-aligned simulation](../professor-aligned/photon-limited-reconstruction-heavy-aligned/README.md) | psnr | 17.9657 | 19.0815 | 1,548,792 |
| Xin Zhang | [Professor-aligned simulation](../professor-aligned/distance-field-segmentation-heavy-aligned/README.md) | dice | 0.2978 | 0.8287 | 1,548,792 |
| Xia Ning | [Public-data companion](../real-data-projects/clinical-risk-missing-labs-heavy-real-data/README.md) | auroc | 0.5819 | 0.5722 | 1,259,354 |
| Lijun Cheng | [Public-data companion](../real-data-projects/cardiac-feature-occlusion-heavy-real-data/README.md) | auroc | 0.8700 | 0.8085 | 681,730 |
| Lubomir Hadjiyski | [Public-data companion](../real-data-projects/mammography-analogue-radiomics-heavy-real-data/README.md) | auroc | 0.8311 | 0.8254 | 1,524,682 |
| Neil Abernethy | [Public-data companion](../real-data-projects/readmission-clinical-informatics-heavy-real-data/README.md) | auroc | 0.6024 | 0.6024 | 681,730 |
| Shuai Huang | [Public-data companion](../real-data-projects/short-horizon-glucose-risk-heavy-real-data/README.md) | auroc | 0.5799 | 0.5805 | 1,256,554 |
| Marc Niethammer | [Public-data companion](../real-data-projects/affine-registration-pairs-heavy-real-data/README.md) | localization_mae_px | 1.1429 | 0.9837 | 1,526,315 |
| Brent Munsell | [Public-data companion](../real-data-projects/morphometry-followup-heavy-real-data/README.md) | mae | 0.0180 | 0.0160 | 665,089 |
| Sayeed Ghani | [Public-data companion](../real-data-projects/inertial-wireless-loss-heavy-real-data/README.md) | auroc | 1.0000 | 1.0000 | 1,256,954 |
| Arghya Kusum Das | [Public-data companion](../real-data-projects/batched-clinical-learning-heavy-real-data/README.md) | elapsed_seconds | 7.0300 | 3.3026 | 671,746 |
| Yung-Hsiang Lu | [Public-data companion](../real-data-projects/edge-resolution-gating-heavy-real-data/README.md) | accuracy | 0.7188 | 0.7000 | 1,525,322 |
| Chaofan Chen | [Public-data companion](../real-data-projects/learned-image-prototypes-heavy-real-data/README.md) | auroc | 0.7918 | 0.8222 | 1,401,896 |
| Hong Xiao | [Public-data companion](../real-data-projects/temporal-signal-artifacts-heavy-real-data/README.md) | auroc | 0.4846 | 0.4874 | 1,257,354 |
| Dimitris N. Metaxas | [Public-data companion](../real-data-projects/masked-fourier-image-recovery-heavy-real-data/README.md) | psnr | 16.8140 | 19.5807 | 1,549,432 |
| Sharon Xiaolei Huang | [Public-data companion](../real-data-projects/polyp-segmentation-corruption-heavy-real-data/README.md) | dice | 0.4004 | 0.4349 | 1,550,072 |
| Xiaohu Guo | [Public-data companion](../real-data-projects/deformation-landmark-regression-heavy-real-data/README.md) | landmark_error_px | 1.2434 | 0.9465 | 1,528,080 |
| Suresh S. Muknahallipatna | [Public-data companion](../real-data-projects/vessel-segmentation-multiscale-heavy-real-data/README.md) | dice | 0.1767 | 0.2395 | 1,549,432 |
| Ngan Le | [Public-data companion](../real-data-projects/caption-noise-crossmodal-heavy-real-data/README.md) | auroc | 0.7399 | 0.7349 | 2,234,626 |
| Dongming Peng | [Public-data companion](../real-data-projects/vital-transition-forecast-heavy-real-data/README.md) | auroc | 0.7281 | 0.7168 | 1,257,354 |
| Changqing Li | [Public-data companion](../real-data-projects/photon-limited-reconstruction-heavy-real-data/README.md) | psnr | 14.3688 | 15.4165 | 1,548,792 |
| Xin Zhang | [Public-data companion](../real-data-projects/distance-field-segmentation-heavy-real-data/README.md) | dice | 0.1546 | 0.0257 | 1,548,792 |
