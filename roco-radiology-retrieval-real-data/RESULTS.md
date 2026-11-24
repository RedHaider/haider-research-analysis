# Validated radiology retrieval results

| Condition | Baseline mean R@5 | Proposed mean R@5 | Change |
| --- | ---: | ---: | ---: |
| Clean | 0.4813 | 0.5563 | **+0.0750** |
| Caption dropout | 0.3875 | 0.4438 | **+0.0563** |
| Image blur | 0.4688 | 0.4875 | +0.0188 |
| Combined | 0.3938 | 0.4438 | **+0.0500** |
| Three-corruption mean | 0.4167 | 0.4583 | **+0.0417** |

The proposed pipeline trained residual projection adapters with symmetric
contrastive loss across three seeds. No adapter exceeded the frozen-CLIP
validation baseline, so validation correctly retained identity adapters. The
reported test improvement therefore comes from density-aware hubness correction,
not from claiming that unsuccessful training improved the encoder.

Protocol: 200 training rows (160 adapter training, 40 validation), 80 official
test rows, frozen CLIP ViT-B/32 embeddings, caption word dropout, Gaussian image
blur and combined corruption. The dataset contains research-publication figures,
not patient records. This is not a clinical or professor-model claim.

