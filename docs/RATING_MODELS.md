# Predicting gloss ratings: easy within a shape, hard across shapes

Models that predict the median human glossiness rating of the Serrano et al. (2021) images (42,120 images: 520
materials × 9 shapes × 9 illuminations, ratings 1–7).

```powershell
.\.venv\Scripts\python.exe -m scripts.highlight_cues --quantile 0.95 --out runs\highlight_cues\cues_q95.csv
.\.venv\Scripts\python.exe -m scripts.cue_model --cues runs\highlight_cues\cues_q95.csv
.\.venv\Scripts\python.exe -m scripts.rating_ceiling
.\.venv\Scripts\python.exe -m scripts.shape_holdout --manifest data\serrano\manifest_all9.csv --cues runs\highlight_cues\cues_all9_q95.csv --refl runs\highlight_cues\reflection_all9_q95.csv --unet-dir "runs/unet_all9_{g}"
```

## The models

- **Highlight cues.** The gloss map P of Nagata et al. marks highlight candidates: ON-centre Laplacian-of-Gaussian
  responses of the luminance at many scales, half-wave rectified, multiplied pairwise across scales and averaged
  (`src/models/features.py`). The highlight region H is where P exceeds a quantile threshold. From H we take coverage,
  contrast and sharpness (after Marlow, Kim & Anderson 2012; the operational definitions are ours), plus threshold-free
  quantiles of P and of luminance. Boosted trees map the cues to the rating.
- **Reflection cues.** How much the object shows the structure of its surroundings: structure inside the object
  compared with the visible background, and structure outside the highlight region.
- **U-Net** with a scalar head, on RGB or on the 21-channel LoG input.

## Results

Pearson r with the median rating. Material groups are always held out (edited and isotropic versions of a measured
material stay with their source).

| Evaluation | Noise ceiling | Cues | U-Net (RGB) |
|---|---|---|---|
| Sphere, unseen materials | 0.94 | 0.92 | 0.90 |
| Sphere, unseen materials and illuminations | | 0.87 | 0.87 |
| Leave one shape out, mean of 9 | | 0.64 | 0.63 |

- **The ceiling is close.** Split-half reliability of the individual ratings gives a ceiling of r = 0.94 on the sphere.
  Both model families reach about 95% of it, so more cues on one shape are pointless.
- **The LoG input adds nothing for the U-Net** (same accuracy as RGB), but a 1.4k-parameter MLP on pooled LoG responses
  already reaches r 0.83–0.86: most of the rating signal is in the hand-made responses.
- **Illumination must be held out too.** With only nine illuminations, a model can learn one correction per
  illumination. Cues that describe the background looked as if they reproduced the human illumination effect under
  material cross-validation, and failed for unseen illuminations. The U-Net's human-like behaviour across illuminations
  was largely the same memorisation.

## Where the models fail

**Shape.** Leave-one-shape-out, highlight cues vs U-Net (r, and mean predicted rating against the human mean):

| Held-out shape | Cues r | U-Net r | Human mean | Cues mean | U-Net mean |
|---|---|---|---|---|---|
| sphere | 0.80 | 0.85 | 2.39 | 2.69 | 1.75 |
| blob | 0.74 | 0.82 | 2.41 | 3.18 | 3.02 |
| bunny | 0.70 | 0.44 | 2.17 | 3.26 | 4.17 |
| Buddha | 0.73 | 0.60 | 2.15 | 3.20 | 5.15 |
| statuette | 0.71 | 0.34 | 2.05 | 3.24 | 4.59 |
| dragon | 0.27 | 0.41 | 1.79 | 4.49 | 5.58 |

Within a shape the ranking transfers partly; the level does not. Every model rates matte materials on finely detailed
shapes as glossy: a lit ridge between shadowed grooves has the centre-surround profile the detector looks for. Humans
rate these shapes as the least glossy.

**Metals and very glossy materials** are under-predicted by 1.3–1.7 rating points by cues and U-Net alike. The cause is
reflections of the environment rather than highlights: a smooth surface with extended reflections may show only a thin
highlight. Reflection cues cut the error for metallic materials from −1.09 to −0.81 points and raise r under the
material × illumination hold-out from 0.81 to 0.85.

## Limits

- U-Nets are single seeds; hold-out runs use the last epoch (the best epoch would be chosen on the held-out data).
- One Serrano shape fills the frame, so cues that need a visible background are undefined there.
- The attribute ratings (contrast of reflections, sharpness, metallicness) correlate 0.65–0.85 with glossiness; no cue
  was specific to one of them.
