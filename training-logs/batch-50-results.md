# Batch 50 — training and evaluation

Twenty trained experiments across ten professors. Each study uses seeds 13, 37 and 61, with validation-selected checkpoints.

| Professor | Study | Primary metric | Compact | Residual | Residual parameters |
|---|---|---|---:|---:|---:|
| Peter G. Jacobs | [Professor-aligned simulation](../professor-aligned/night-glucose-aligned/README.md) | auroc | 0.8110 | 0.8075 | 37,082 |
| Nikolaos V. Tsekos | [Professor-aligned simulation](../professor-aligned/image-localization-aligned/README.md) | mean_error_px | 9.0660 | 3.1450 | 99,650 |
| Jens Gregor | [Professor-aligned simulation](../professor-aligned/sparse-view-reconstruction-aligned/README.md) | psnr_db | 16.6776 | 16.8724 | 109,921 |
| Shandong Wu | [Professor-aligned simulation](../professor-aligned/breast-shift-aligned/README.md) | auroc | 0.9999 | 1.0000 | 99,218 |
| Parisa Rashidi | [Professor-aligned simulation](../professor-aligned/wearable-sleep-transition-aligned/README.md) | auroc | 0.6974 | 0.6953 | 37,010 |
| Jiaqi Gong | [Professor-aligned simulation](../professor-aligned/activity-temporal-aligned/README.md) | accuracy | 0.5678 | 0.9338 | 37,614 |
| Peter Sadowski | [Professor-aligned simulation](../professor-aligned/histology-tissue-aligned/README.md) | accuracy | 0.2926 | 0.9003 | 100,329 |
| Bobak J. Mortazavi | [Professor-aligned simulation](../professor-aligned/wearable-missing-stream-aligned/README.md) | accuracy | 0.6136 | 0.8252 | 37,614 |
| Tolga Tasdizen | [Professor-aligned simulation](../professor-aligned/retinal-vessel-context-aligned/README.md) | dice | 0.9292 | 0.9793 | 109,921 |
| Luciano Prevedello | [Professor-aligned simulation](../professor-aligned/calibrated-radiograph-aligned/README.md) | brier | 0.0214 | 0.0000 | 99,218 |
| Peter G. Jacobs | [Public-data companion](../real-data-projects/night-glucose-real-data/README.md) | auroc | 0.5080 | 0.5179 | 37,082 |
| Nikolaos V. Tsekos | [Public-data companion](../real-data-projects/image-localization-real-data/README.md) | mean_error_px | 10.0902 | 7.4234 | 99,650 |
| Jens Gregor | [Public-data companion](../real-data-projects/sparse-view-reconstruction-real-data/README.md) | psnr_db | 15.2119 | 14.6350 | 109,921 |
| Shandong Wu | [Public-data companion](../real-data-projects/breast-shift-real-data/README.md) | auroc | 0.7549 | 0.8042 | 99,218 |
| Parisa Rashidi | [Public-data companion](../real-data-projects/wearable-sleep-transition-real-data/README.md) | auroc | 0.5691 | 0.5650 | 37,010 |
| Jiaqi Gong | [Public-data companion](../real-data-projects/activity-temporal-real-data/README.md) | accuracy | 0.6173 | 0.6503 | 37,614 |
| Peter Sadowski | [Public-data companion](../real-data-projects/histology-tissue-real-data/README.md) | accuracy | 0.6808 | 0.6910 | 100,329 |
| Bobak J. Mortazavi | [Public-data companion](../real-data-projects/wearable-missing-stream-real-data/README.md) | accuracy | 0.6110 | 0.6503 | 37,614 |
| Tolga Tasdizen | [Public-data companion](../real-data-projects/retinal-vessel-context-real-data/README.md) | dice | 0.2356 | 0.3784 | 109,921 |
| Luciano Prevedello | [Public-data companion](../real-data-projects/calibrated-radiograph-real-data/README.md) | brier | 0.1162 | 0.1491 | 99,218 |
