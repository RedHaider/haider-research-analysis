# ROCO Radiology Image-Text Retrieval Benchmark

Real-data companion to the professor-aligned multimodal patient-retrieval project
for Joyce Yan-Ran Wang. It uses real radiology figures and captions from the
public ROCOv2-derived Radiology_mini dataset.

The benchmark compares frozen CLIP retrieval with validation-gated residual
projection heads and density-aware hubness correction. It evaluates clean retrieval and
robustness to caption word dropout, image blur and combined corruption using
Recall@1/5/10 and mean reciprocal rank across three training seeds.

All Hugging Face data/model caches and extracted embeddings are kept on drive D.
This is a portfolio retrieval benchmark, not a clinical patient-search system or
a reproduction of Professor Wang's published model.

Data source: [`unsloth/Radiology_mini`](https://huggingface.co/datasets/unsloth/Radiology_mini),
a 2,305-row subset derived from ROCOv2. This experiment uses 200 official training
rows and 80 official test rows under the dataset's CC BY-NC 2.0 license.

## Validated results

Clean bidirectional Recall@5 increased from **0.4813 to 0.5563** (+0.0750),
while clean mean reciprocal rank increased from 0.3637 to 0.3941. Across caption
dropout, image blur and combined corruption, mean Recall@5 increased from
**0.4167 to 0.4583** (+0.0417). No trained adapter exceeded the validation
baseline, so the identity adapter was retained; the defensible improvement comes
from density-aware hubness correction rather than an unsupported training claim.

