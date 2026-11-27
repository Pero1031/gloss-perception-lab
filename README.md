# gloss-perception-lab

Image-computable models of perceived gloss, tested against the human ratings of Serrano et al. (2021):

- **Cue models**: highlight cues (coverage, contrast, sharpness) and reflection cues, fed to a small regressor.
- **U-Nets** on RGB or on multi-scale centre-surround (LoG) responses.
- **Highlight maps learnt from privileged information**: a monocular U-Net trained on rendered scenes to predict the
  left-right stereo residual or the rendered specular share.

## Findings

Pearson r between model and median human glossiness rating.

| Evaluation | Noise ceiling | Cues | U-Net (RGB) | Stereo-residual map | Specular map |
|---|---|---|---|---|---|
| Sphere, unseen materials | 0.94 | 0.92 | 0.90 | | |
| Sphere, unseen materials and lights | | 0.87 | 0.87 | | |
| Leave one shape out (mean of 9) | | 0.64 | 0.63 | 0.51 | 0.42 |

- Within one shape, cues and U-Nets reach about 95% of the noise ceiling. Generalisation across shapes is the open
  problem.
- Every model rates matte materials on finely detailed shapes as glossy. Centre-surround detectors respond to lit
  ridges between shadowed grooves as if they were highlights.
- Metals and very glossy materials are under-predicted because of reflections of the environment, not highlights.
  Reflection cues reduce this error.
- Stereo- or render-supervised highlight maps are calibrated on renders but transfer worse to Serrano than simple cues
  (`notes/handoff_stereo_highlight.md`).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Rendering scripts need Blender 4.5; `scripts/stereo_matching.py` needs `opencv-python`.

## Data

Serrano et al. (2021): 42,120 images (520 materials x 9 shapes x 9 illuminations) with median 1-7 ratings, from
https://mig.mpi-inf.mpg.de/. Put `Serrano_SIGG2021_PerceptualRatings.xlsx` in `data/serrano/`, then fetch images per
shape directly from the remote zip:

```powershell
.\.venv\Scripts\python.exe -I scripts\fetch_serrano_images.py --geometry sphere --size 256 --workers 8
.\.venv\Scripts\python.exe -I scripts\make_serrano_manifest.py --image-root data\serrano\images --out data\serrano\manifest_sphere.csv
```

`data/`, `runs/` and `docs/` are not tracked.

## Code

```
src/models/   U-Net (scalar or map head), LoG input features and gloss map P, TinyGlossNet
src/data/     manifest dataset with material-group split
src/train.py  training with rating-balanced sampling and material / illumination / shape hold-out
scripts/      cue extraction, evaluations, rendering and figures
```

```powershell
.\.venv\Scripts\python.exe -m src.train --dataset synthetic --epochs 5   # smoke test
.\.venv\Scripts\python.exe -m src.train --dataset manifest --manifest data\serrano\manifest_sphere.csv --root data\serrano\images --size 128
```

| Scripts | Purpose |
|---|---|
| `highlight_cues`, `reflection_cues` | cue extraction |
| `cue_model`, `cue_analysis`, `reflection_eval`, `rating_ceiling` | cue-model evaluation, error analysis, noise ceiling |
| `cue_illum_holdout`, `unet_illum_holdout` | material x illumination hold-out |
| `shape_masks`, `shape_holdout`, `shape_figures` | leave-one-shape-out |
| `stereo_render`, `stereo_residual`, `stereo_matching` | stereo-residual pilot (Blender) |
| `stereo_dataset`, `stereo_targets`, `train_highlight`, `serrano_highlight_eval` | training and evaluating highlight-map networks |
| `serrano_shape_render`, `serrano_render_eval`, `data_overview`, `false_highlight_figure` | diagnosis of the transfer failure |
| `merl_fit` | Principled BSDF fits of the MERL BRDFs |

Scripts that import `bpy` run inside Blender (`blender -b --factory-startup -P <script> -- <args>`). The others run as
`python -m scripts.<name>`.

## References

- Serrano, A., Chen, B., Wang, C., Piovarči, M., Seidel, H.-P., Didyk, P., Myszkowski, K. (2021). The effect of shape and
  illumination on material perception: model and applications. *ACM Transactions on Graphics*, 40(4).
- Marlow, P. J., Kim, J., Anderson, B. L. (2012). The perception and misperception of specular surface reflectance.
  *Current Biology*, 22(20).
- Prokott, K. E., Fleming, R. W. (2022). Identifying specular highlights: insights from deep learning. *Journal of
  Vision*, 22(7).
- Morimoto et al. (2025). bioRxiv 2025.05.09.653112.
