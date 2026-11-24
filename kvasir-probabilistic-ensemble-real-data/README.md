# Kvasir Probabilistic Lesion Ensemble

Real-data companion for Abdullah-Al-Zubaer Imran. It trains a deterministic
pixel classifier and a bootstrap Extra-Trees ensemble on real Kvasir-SEG images
and masks, comparing Dice, Brier score and uncertainty concentration at lesion
boundaries. Images and masks remain on drive D.

This is a classical probabilistic segmentation benchmark, not a clinical system
or a reproduction of the professor's model.

Validated three-seed result: the five-member ensemble improved held-out Dice
from **0.4058 to 0.4700** (+0.0642) and reduced Brier score from **0.2001 to
0.1375**. Absolute Dice remains modest because this is a deliberately lightweight
64-pixel classical model rather than a deep clinical segmentation network.

