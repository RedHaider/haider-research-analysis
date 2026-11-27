# Batch 79 — larger neural studies

38 studies across 19 professors. Each result averages seeds 13 and 37. Experiments ran locally on CPU with validation-selected models and kept the original training and evaluation partitions.

| Professor | Study | Metric | Compact | Larger | Larger parameters | Input dimensions |
|---|---|---|---:|---:|---:|---|
| Ping Zhang | [Professor-aligned simulation](../professor-aligned/history-mask-readmission-heavy-aligned/README.md) | auroc | 0.7962 | 0.7920 | 673,282 | 15 |
| Qing Wu | [Professor-aligned simulation](../professor-aligned/held-hospital-cardiac-heavy-aligned/README.md) | auroc | 0.7783 | 0.7769 | 681,730 | 26 |
| Meliha Yetisgen | [Professor-aligned simulation](../professor-aligned/biomedical-caption-nlp-heavy-aligned/README.md) | auroc | 0.9791 | 0.9831 | 1,448,194 | 1024 |
| Daiwei (David) Zhang | [Professor-aligned simulation](../professor-aligned/histology-stain-variation-heavy-aligned/README.md) | accuracy | 1.0000 | 1.0000 | 1,527,793 | 3 × 36 × 36 |
| May Dongmei Wang | [Professor-aligned simulation](../professor-aligned/clinical-data-ablation-heavy-aligned/README.md) | auroc | 0.8222 | 0.8387 | 1,258,954 | 16 × 24 |
| Khan Iftekharuddin | [Professor-aligned simulation](../professor-aligned/brain-tumor-image-classifier-heavy-aligned/README.md) | auroc | 1.0000 | 1.0000 | 1,524,042 | 1 × 64 × 64 |
| Xintao Wu | [Professor-aligned simulation](../professor-aligned/age-balanced-readmission-heavy-aligned/README.md) | auroc | 0.7962 | 0.7685 | 673,282 | 15 |
| Xin (Shane) Li | [Professor-aligned simulation](../professor-aligned/dense-motion-field-heavy-aligned/README.md) | endpoint_error_px | 9.4329 | 9.2690 | 1,550,088 | 2 × 40 × 40 |
| Guy C. Hembroff | [Professor-aligned simulation](../professor-aligned/wearable-sleep-hr-heavy-aligned/README.md) | auroc | 0.9920 | 0.9970 | 1,256,354 | 3 × 10 |
| John W. Sheppard | [Professor-aligned simulation](../professor-aligned/survival-risk-ranking-heavy-aligned/README.md) | auroc | 0.6794 | 0.6873 | 672,514 | 14 |
| MD Shaad Mahmud | [Professor-aligned simulation](../professor-aligned/physio-band-limited-heavy-aligned/README.md) | auroc | 1.0000 | 1.0000 | 1,257,354 | 8 × 128 |
| Fusheng Wang | [Professor-aligned simulation](../professor-aligned/regional-histology-spatial-heavy-aligned/README.md) | accuracy | 1.0000 | 1.0000 | 1,529,073 | 5 × 32 × 32 |
| Safwan Wshah | [Professor-aligned simulation](../professor-aligned/radiology-modality-robust-heavy-aligned/README.md) | auroc | 0.9995 | 0.9990 | 2,234,626 | 2048 |
| Ulugbek S. Kamilov | [Professor-aligned simulation](../professor-aligned/frequency-consistent-inverse-heavy-aligned/README.md) | psnr | 20.0781 | 23.9796 | 1,549,432 | 2 × 64 × 64 |
| Assefaw H. Gebremedhin | [Professor-aligned simulation](../professor-aligned/cohort-size-readmission-heavy-aligned/README.md) | auroc | 0.7869 | 0.7949 | 673,282 | 15 |
| Tony Luo | [Professor-aligned simulation](../professor-aligned/robust-breast-transfer-heavy-aligned/README.md) | auroc | 1.0000 | 1.0000 | 1,524,042 | 1 × 32 × 32 |
| Jianjun Hu | [Professor-aligned simulation](../professor-aligned/retina-severity-net-heavy-aligned/README.md) | auroc | 1.0000 | 1.0000 | 1,525,322 | 3 × 32 × 32 |
| Weimin Zhou | [Professor-aligned simulation](../professor-aligned/quality-observer-heavy-aligned/README.md) | accuracy | 1.0000 | 0.9976 | 1,524,395 | 1 × 64 × 64 |
| Neal Bangerter | [Professor-aligned simulation](../professor-aligned/mri-accelerated-denoising-heavy-aligned/README.md) | psnr | 21.3309 | 25.2835 | 1,548,792 | 1 × 64 × 64 |
| Ping Zhang | [Public-data companion](../real-data-projects/history-mask-readmission-heavy-real-data/README.md) | auroc | 0.6132 | 0.6222 | 671,746 | 13 |
| Qing Wu | [Public-data companion](../real-data-projects/held-hospital-cardiac-heavy-real-data/README.md) | auroc | 0.8700 | 0.8085 | 681,730 | 26 |
| Meliha Yetisgen | [Public-data companion](../real-data-projects/biomedical-caption-nlp-heavy-real-data/README.md) | auroc | 0.5440 | 0.3464 | 1,448,194 | 1024 |
| Daiwei (David) Zhang | [Public-data companion](../real-data-projects/histology-stain-variation-heavy-real-data/README.md) | accuracy | 0.6348 | 0.6098 | 1,527,793 | 3 × 36 × 36 |
| May Dongmei Wang | [Public-data companion](../real-data-projects/clinical-data-ablation-heavy-real-data/README.md) | auroc | 0.5819 | 0.5715 | 1,259,354 | 18 × 24 |
| Khan Iftekharuddin | [Public-data companion](../real-data-projects/brain-tumor-image-classifier-heavy-real-data/README.md) | auroc | 0.7570 | 0.7400 | 1,524,042 | 1 × 64 × 64 |
| Xintao Wu | [Public-data companion](../real-data-projects/age-balanced-readmission-heavy-real-data/README.md) | auroc | 0.6132 | 0.6145 | 671,746 | 13 |
| Xin (Shane) Li | [Public-data companion](../real-data-projects/dense-motion-field-heavy-real-data/README.md) | endpoint_error_px | 9.3970 | 8.7577 | 1,550,088 | 2 × 40 × 40 |
| Guy C. Hembroff | [Public-data companion](../real-data-projects/wearable-sleep-hr-heavy-real-data/README.md) | auroc | 0.6388 | 0.6106 | 1,256,354 | 3 × 8 |
| John W. Sheppard | [Public-data companion](../real-data-projects/survival-risk-ranking-heavy-real-data/README.md) | auroc | 0.7836 | 0.7253 | 672,514 | 14 |
| MD Shaad Mahmud | [Public-data companion](../real-data-projects/physio-band-limited-heavy-real-data/README.md) | auroc | 0.4846 | 0.4906 | 1,257,354 | 8 × 128 |
| Fusheng Wang | [Public-data companion](../real-data-projects/regional-histology-spatial-heavy-real-data/README.md) | accuracy | 0.5076 | 0.4696 | 1,529,073 | 5 × 32 × 32 |
| Safwan Wshah | [Public-data companion](../real-data-projects/radiology-modality-robust-heavy-real-data/README.md) | auroc | 0.7399 | 0.7234 | 2,234,626 | 2048 |
| Ulugbek S. Kamilov | [Public-data companion](../real-data-projects/frequency-consistent-inverse-heavy-real-data/README.md) | psnr | 16.9560 | 19.3603 | 1,549,432 | 2 × 64 × 64 |
| Assefaw H. Gebremedhin | [Public-data companion](../real-data-projects/cohort-size-readmission-heavy-real-data/README.md) | auroc | 0.5756 | 0.6158 | 671,746 | 13 |
| Tony Luo | [Public-data companion](../real-data-projects/robust-breast-transfer-heavy-real-data/README.md) | auroc | 0.8328 | 0.7840 | 1,524,042 | 1 × 32 × 32 |
| Jianjun Hu | [Public-data companion](../real-data-projects/retina-severity-net-heavy-real-data/README.md) | auroc | 0.7872 | 0.7803 | 1,525,322 | 3 × 32 × 32 |
| Weimin Zhou | [Public-data companion](../real-data-projects/quality-observer-heavy-real-data/README.md) | accuracy | 0.9967 | 0.9946 | 1,524,395 | 1 × 64 × 64 |
| Neal Bangerter | [Public-data companion](../real-data-projects/mri-accelerated-denoising-heavy-real-data/README.md) | psnr | 17.7579 | 20.7034 | 1,548,792 | 1 × 64 × 64 |
