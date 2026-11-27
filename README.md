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

## Gloss manipulation (in progress)

Goal: suppress or enhance the gloss of an object in an image. Edits are scored without observers, on renders whose
specular pass is known: an edit is compared with the same edit applied to the true specular component (RMSE x100 in
sRGB, lower is better; 48 renders of Serrano geometries and 400 random scenes).

Suppression of glossy objects:

| Method | Serrano geometries | Random scenes |
|---|---|---|
| No edit | 8.6-10.7 | 11-12 |
| Highlight layer over the region where the gloss map P is high | 9.7-14.3 | 15 |
| Same, region from the stereo-residual network | 7.4-8.6 | 9-10 |
| Per-pixel specular share from the specular-share network | 5.1-18.0 | 6 |
| TSHRNet (Fu et al. 2023), mixed-data weights | 6.1-6.6 | 11 |
| Per-pixel specular share, true values (upper bound) | 1.3-1.6 | 1.5 |

- The gloss map P finds compact bright highlights (pixel AUC 0.87-0.95) but not the specular component as a whole
  (AUC 0.53-0.63), which on glossy objects is a reflection of the environment spread over the surface. Editing a
  highlight layer in the region flagged by P is worse than no edit and damages matte objects.
- Scaling a per-pixel specular share is the right edit model (last row). The share network works on images like its
  training set and fails on finely detailed geometry, where it calls matte objects glossy. Its training set (noise
  blobs, eight environment maps) is too narrow.
- TSHRNet removes compact highlights and leaves broad reflections.
- Planned pipeline: estimate the amount of specular reflection first. Glossy input: scale the specular part. Matte
  input: leave it alone when suppressing; enhancement has to synthesise highlights.
- TSHRNet is built for images that have highlights; matte inputs are outside its scope.

Adding gloss to a matte object (`highlight_synthesis`, true surface normals for now). The layer reflects nothing of
the real surroundings. It is computed from the normals and from the lighting implied by a Lambert fit to the shading,
`Y ~ n.L + b`: the light direction (combined with a weak prior that light comes from above), a uniform part and a
directional part.

- Highlights alone do not look glossy. If only bright spots are added, the rest of the surface keeps its matte
  luminance and the spots read as white paint. The layer needs a wide component over the whole lit side as well, as a
  smooth surface under a strong light would show.
- The wide component follows from the fit (a mirror shows the fitted lighting, weighted by Fresnel reflectance); the
  narrow one is a GGX lobe of a compact source. With the true albedo, the specular share of the object's light matches
  the real semi-glossy renders across shapes and lights (ours 0.23-0.53, real 0.21-0.47, same order).
- Not from the image: the albedo, which sets the strength, and how much of the directional light is one compact source.
- Open: structure inside reflections on smooth surfaces, perceptual ratings.

Normals from one image (`normal_estimation`; 12 dark matte renders at 512 px, mean over four shapes):

| Estimator | Normal error | Shape error (smoothed) | Light direction error | Wide layer r | s / image |
|---|---|---|---|---|---|
| MoGe-2 ViT-L | 13.4 deg | 9.2 deg | 6.2 deg | 0.79 | 3.3 |
| MoGe-2 ViT-B | 14.1 deg | 9.4 deg | 8.3 deg | 0.76 | 1.5 |
| Marigold-Normals v1-1 | 24.4 deg | 18.1 deg | 21.6 deg | 0.56 | 1.9-3.5 |
| Silhouette inflation | 27.5 deg | 21.8 deg | 18.5 deg | 0.42 | 2.3 |
| Silhouette inflation + shading step | 25.1 deg | 19.7 deg | 21.3 deg | 0.56 | 2.3 |

"Wide layer r" is the correlation between the wide part of the specular layer computed from estimated normals and
from true normals. The wide part survives estimated normals; the position of narrow highlights does not (whole-layer
r 0.3-0.4 for every estimator). Inflation is adequate only for the sphere. Times are on a Quadro M4000.

On photographs (`photo_gloss`): object region from a mask or a hue range, normals and view directions from MoGe-2,
then the layer above. Compositing that keeps the object's identity: the wide part changes luminance only (the pixel
keeps its chromaticity), only the narrow highlight is added as white, and the diffuse part is scaled so that the
object's mean luminance stays the same. Adding the whole layer as white light brightens and desaturates the object.
The albedo, which sets the strength, is still a guess.

| Script | Purpose |
|---|---|
| `gloss_edit` | detectors (P and variants), specular layer and share edits, additive edit |
| `gloss_edit_eval` | detection and edit errors against the true specular pass |
| `tshrnet_eval` | the same test for TSHRNet (code and weights from its authors, placed in `external/`) |
| `highlight_synthesis` | specular layers for matte objects from normals and the shading fit, compared with real renders |
| `normal_estimation` | normals from one image (silhouette inflation, MoGe-2, Marigold) scored by normal error and by the specular layer |
| `photo_gloss` | the whole chain on a photograph |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Rendering scripts need Blender 4.5; `scripts/stereo_matching.py` needs `opencv-python`.
`normal_estimation` and `photo_gloss` also need `diffusers`, `transformers`, `huggingface_hub`,
`opencv-python-headless` and the MoGe repository (https://github.com/microsoft/MoGe) cloned into `external/MoGe`
with its `utils3d_moge` dependency; model weights are fetched from the Hugging Face hub on first use.

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
- Wang, R. et al. (2025). MoGe-2: accurate monocular geometry with metric scale and sharp details.
- Ke, B. et al. (2025). Marigold: affordable adaptation of diffusion-based image generators for image analysis.
- Fu, G., Zhang, Q., Zhu, L., Xiao, C., Li, P. (2023). Towards high-quality specular highlight removal by leveraging
  large-scale synthetic data. *ICCV*.
