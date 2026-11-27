# Editing existing gloss: scale the specular share, not a highlight region

How to suppress or enhance the gloss an object already has. Edits are scored without observers, on renders whose
specular pass is known.

```powershell
.\.venv\Scripts\python.exe -m scripts.gloss_edit_eval                    # all specular pixels count as positives
.\.venv\Scripts\python.exe -m scripts.gloss_edit_eval --bright 0.3 --out runs\gloss_edit_bright   # bright highlights only
.\.venv\Scripts\python.exe -m scripts.tshrnet_eval                       # needs external/TSHRNet and its weights
```

## Test sets and score

- **A**: 48 renders of four Serrano geometries (sphere, bunny, Buddha, dragon) with our own matte, semi-glossy and
  glossy materials under three Serrano environment maps (`serrano_shape_render.py`).
- **B**: 400 held-out random scenes of the stereo training set (`stereo_dataset.py`).

An edit removes the specular part (gain 0) or doubles it (gain 2). It is compared with the same edit applied to the
true specular component: RMSE × 100 in sRGB over object pixels. Lower is better; "no edit" is the reference.

## Detection: the gloss map finds highlights, not the specular component

Pixel AUC for specular pixels, and the share of a matte object flagged at the threshold that finds half of them.

| Detector | AUC, all specular pixels | AUC, bright highlights | Matte dragon flagged (bright) |
|---|---|---|---|
| Gloss map P | 0.63 / 0.53 | 0.95 / 0.87 | 17% |
| P × local skewness of luminance | 0.60 / 0.56 | 0.87 / 0.83 | 2.6% |
| P × ON/OFF balance | 0.43 / 0.36 | 0.81 / 0.75 | 18% |
| Stereo-residual network | 0.81 / 0.93 | 0.91 / 0.91 | 0.8% |

Pairs are set A / set B. On a glossy object most of the specular light is a reflection of the surroundings spread over
the surface. P responds to compact bright peaks only. Of the hand-made rejection rules for matte ridges, only local
skewness helped; comparing ON and OFF responses made things worse.

## Edits

Suppression (gain 0) of glossy objects:

| Method | Set A | Set B |
|---|---|---|
| No edit | 8.6–10.7 | 11–12 |
| Additive rule I − gain · P, best gain | 7.9–10.2 | 11–12 |
| Highlight layer over the region where P is high | 9.7–14.3 | 15 |
| Same, region from the stereo-residual network | 7.4–8.6 | 9–10 |
| TSHRNet (Fu et al. 2023), mixed-data weights | 6.1–6.6 | 11 |
| Per-pixel specular share from the share network | 5.1–18.0 | 6 |
| Per-pixel specular share, true values | 1.3–1.6 | 1.5 |

- **A region-and-layer model is the wrong decomposition.** Even with the true region, splitting the image into a smooth
  base and a highlight layer leaves an error of 7–10: the specular part is not a local blob on a smooth base.
- **Scaling a per-pixel specular share is right**, as the last row shows. What is missing is a good estimate of the
  share.
- **The share network fails outside its training set.** It was trained on noise-displaced blobs under eight
  environment maps. On matte renders of the Buddha and the dragon it predicts a specular share of 0.26–0.61 (truth 0),
  uniformly over the object. Halving the object's size in the frame removes most of the error, a dark albedo doubles it:
  visible fine relief on a dark surface is read as gloss. The training set is too narrow in shape and lighting.
- **TSHRNet** is built to remove highlights from images that have them. It leaves broad reflections in place, and matte
  inputs are outside its scope (it alters them).
- With P as the detector, the layer edit damages matte objects (error 13–28 where no edit is 0.1).

## Consequences

Estimate the amount of specular reflection first. A glossy input is edited by scaling its specular part. A matte input
is left alone when suppressing, and enhancement has to synthesise the specular layer
([GLOSS_SYNTHESIS.md](GLOSS_SYNTHESIS.md)).
