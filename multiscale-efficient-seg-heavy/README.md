# Efficient multiscale segmentation (simulation)

Dice for dense vs multiscale decoder was 0.819 vs 0.823 across three seeds. Parameters: 15,353 vs 7,393.

![Mean held-out scores](comparison.svg)

## Study design

Compare a dense segmentation decoder with an original efficient decoder using parallel 3×3 and 5×5 depthwise convolutions and channel gating. Both use the same encoder, skip fusion, BCE plus soft Dice objective and validation-based epoch selection. Train from scratch for nine epochs for each of three seeds. The experiment takes the multiscale depthwise and attention ideas from EMCAD and uses a smaller custom architecture. Generate 480 RGB phantoms with lesion masks, background texture and variable contrast per seed.

Independent portfolio experiment inspired by work from Md Mostafijur Rahman. [Research source](https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html)

## Results

| Method | dice | parameters | test_images |
|---|---:|---:|---:|
| dense | 0.8186 | 15353.0000 | 86.0000 |
| depthwise | 0.8229 | 7393.0000 | 86.0000 |

Per-seed results are in [results.json](results.json).

## Reproduce

Local CPU run, Python 3.10. Model checkpoints are generated during training. Set PORTFOLIO_DATA to reuse an existing data directory.

```sh
python -m pip install -r requirements.txt
python train.py
```

[Training notebook](reproduce.ipynb)
