# Normals from one image: MoGe-2 is good enough for the sheen, not for highlight positions

The specular layer of [GLOSS_SYNTHESIS.md](GLOSS_SYNTHESIS.md) needs surface normals. This page compares ways to get
them from a single image of a matte object.

```powershell
blender -b --factory-startup -P scripts/serrano_shape_render.py -- --out data/serrano_render_512 --size 512 --materials matte_dark,semi_blue
.\.venv\Scripts\python.exe -m scripts.normal_estimation        # -> runs/normals/summary.txt, normals.png, synthesis.png
```

Needs `diffusers`, `transformers`, `huggingface_hub`, `opencv-python-headless`, and the MoGe repository
(https://github.com/microsoft/MoGe) in `external/MoGe` with its `utils3d_moge` dependency. Weights are fetched from
the Hugging Face hub on first use.

## Test

Twelve renders at 512 px: four Serrano geometries with a dark matte material under three environment maps. True
normals come from the position pass. Each estimator is given the tone-mapped image with its background.

- **Silhouette inflation**: normals point outwards on the contour and are diffused into the interior. No network.
- **Inflation + shading step**: each normal is then moved along the fitted light until it reproduces the shading.
- **MoGe-2** (Wang et al. 2025), ViT-B and ViT-L normal models, feed-forward.
- **Marigold-Normals v1-1** (Ke et al. 2025), Stable Diffusion based, 1 and 4 steps.

## Results

Mean over the four shapes. Shape error: angular error after smoothing both normal maps (σ 8 px). Light error: error of
the light direction fitted to the shading. Layer r: correlation of the specular layer from estimated normals with the
layer from true normals, for its wide part and as a whole.

| Estimator | Normal error | Shape error | Relief r | Light error | Wide layer r | Whole layer r | s / image |
|---|---|---|---|---|---|---|---|
| MoGe-2 ViT-L | 13.4° | 9.2° | 0.62 | 6.2° | 0.79 | 0.31 | 3.3 |
| MoGe-2 ViT-B | 14.1° | 9.4° | 0.58 | 8.3° | 0.76 | 0.34 | 1.5 |
| Marigold, 4 steps | 24.4° | 18.2° | 0.38 | 21.8° | 0.58 | 0.24 | 3.5 |
| Inflation | 27.5° | 21.8° | 0.22 | 18.5° | 0.42 | 0.25 | 2.3 |
| Inflation + shading step | 25.1° | 19.7° | 0.40 | 21.3° | 0.56 | 0.31 | 2.3 |
| MoGe-2 ViT-L + shading step | 14.7° | 10.4° | 0.58 | 8.9° | 0.78 | 0.38 | |

Times on a Quadro M4000 (8 GB); memory 1.5 GB (ViT-B), 2.4 GB (ViT-L), 6 GB (Marigold).

- **MoGe-2 is best on every measure**, and keeps the fine relief (r 0.75–0.83 on the bunny, Buddha and dragon).
- **The wide part survives estimated normals** (r about 0.8). The whole layer does not (0.3–0.4 for every estimator):
  a narrow highlight moves with a few degrees of normal error.
- **Inflation works only for the sphere** (11° there, 31–37° elsewhere). Normals on the contour point fully sideways,
  so Fresnel reflectance draws a white rim around the object.
- **The shading step** restores some relief on inflated normals (0.22 → 0.40) and does not improve MoGe-2.
- **Marigold fails on the smooth sphere** (about 30°) and is behind MoGe-2 on the other shapes.
- Axis conventions differ: MoGe-2 uses an OpenCV camera (y down, z forward), so its normals are multiplied by
  (1, −1, −1) to get x right, y up, z towards the viewer. The script finds the signs on the sphere.

On a photograph (grapes, [GLOSS_SYNTHESIS.md](GLOSS_SYNTHESIS.md)) MoGe-2 resolved each grape as a rounded surface.

## Limits

- Renders, and one photograph without ground truth.
- View directions come from the true camera on the renders, and from MoGe-2's point map on photographs.
- Published benchmarks do not settle the choice: recent estimators are within 1–2° of each other on indoor scenes, and
  none is evaluated on single objects. No study gives the normal accuracy a synthesised highlight needs.
- None of these runs in real time on this GPU.
