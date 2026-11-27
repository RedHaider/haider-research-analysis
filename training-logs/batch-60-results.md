# Batch 60 — larger neural benchmarks

Twenty trained studies for ten professors. Both arms use two seeds, local CPU, and validation-selected checkpoints. The larger arm can train up to 12 epochs and the compact arm up to eight.

| Professor | Study | Metric | Compact | Larger | Larger parameters | Input dimensions |
|---|---|---|---:|---:|---:|---|
| Kimerly Powell | [Professor-aligned simulation](../professor-aligned/microscopy-restoration-heavy-aligned/README.md) | psnr | 23.1314 | 27.1024 | 1,551,384 | 3 × 64 × 64 |
| Chuan Zhou | [Professor-aligned simulation](../professor-aligned/image-caption-interaction-heavy-aligned/README.md) | auroc | 0.9995 | 0.9992 | 2,234,626 | 2048 |
| Naleef Fareed | [Professor-aligned simulation](../professor-aligned/readmission-disparity-heavy-aligned/README.md) | auroc | 0.7962 | 0.7926 | 673,282 | 15 |
| Shamim Nemati | [Professor-aligned simulation](../professor-aligned/icu-pressure-trajectory-heavy-aligned/README.md) | auroc | 0.7917 | 0.7935 | 1,258,954 | 16 × 24 |
| Rahmat Beheshti | [Professor-aligned simulation](../professor-aligned/admission-length-risk-heavy-aligned/README.md) | auroc | 0.7654 | 0.7646 | 673,282 | 15 |
| Mingchen Gao | [Professor-aligned simulation](../professor-aligned/polyp-boundary-model-heavy-aligned/README.md) | dice | 0.6742 | 0.8591 | 1,550,072 | 3 × 64 × 64 |
| Ali Cafer Gurbuz | [Professor-aligned simulation](../professor-aligned/undersampled-xray-reconstruction-heavy-aligned/README.md) | psnr | 24.0790 | 25.8996 | 1,549,432 | 2 × 64 × 64 |
| M. Salman Asif | [Professor-aligned simulation](../professor-aligned/motion-image-restoration-heavy-aligned/README.md) | psnr | 24.3119 | 25.5289 | 1,548,792 | 1 × 64 × 64 |
| Chen Liang | [Professor-aligned simulation](../professor-aligned/clinical-view-fusion-heavy-aligned/README.md) | auroc | 0.8222 | 0.8386 | 1,258,954 | 16 × 24 |
| Diane J. Cook | [Professor-aligned simulation](../professor-aligned/inertial-activity-drift-heavy-aligned/README.md) | accuracy | 0.9865 | 0.9743 | 1,258,366 | 6 × 64 |
| Kimerly Powell | [Public-data companion](../real-data-projects/microscopy-restoration-heavy-real-data/README.md) | psnr | 19.8719 | 24.2097 | 1,551,384 | 3 × 64 × 64 |
| Chuan Zhou | [Public-data companion](../real-data-projects/image-caption-interaction-heavy-real-data/README.md) | auroc | 0.7399 | 0.7430 | 2,234,626 | 2048 |
| Naleef Fareed | [Public-data companion](../real-data-projects/readmission-disparity-heavy-real-data/README.md) | auroc | 0.6132 | 0.6294 | 671,746 | 13 |
| Shamim Nemati | [Public-data companion](../real-data-projects/icu-pressure-trajectory-heavy-real-data/README.md) | auroc | 0.6693 | 0.6696 | 1,258,954 | 16 × 24 |
| Rahmat Beheshti | [Public-data companion](../real-data-projects/admission-length-risk-heavy-real-data/README.md) | auroc | 0.5652 | 0.5660 | 667,138 | 7 |
| Mingchen Gao | [Public-data companion](../real-data-projects/polyp-boundary-model-heavy-real-data/README.md) | dice | 0.3327 | 0.4154 | 1,550,072 | 3 × 64 × 64 |
| Ali Cafer Gurbuz | [Public-data companion](../real-data-projects/undersampled-xray-reconstruction-heavy-real-data/README.md) | psnr | 25.6014 | 27.0607 | 1,549,432 | 2 × 64 × 64 |
| M. Salman Asif | [Public-data companion](../real-data-projects/motion-image-restoration-heavy-real-data/README.md) | psnr | 26.2952 | 27.7885 | 1,548,792 | 1 × 64 × 64 |
| Chen Liang | [Public-data companion](../real-data-projects/clinical-view-fusion-heavy-real-data/README.md) | auroc | 0.5819 | 0.5668 | 1,259,354 | 18 × 24 |
| Diane J. Cook | [Public-data companion](../real-data-projects/inertial-activity-drift-heavy-real-data/README.md) | accuracy | 0.6310 | 0.7090 | 1,258,366 | 6 × 64 |
