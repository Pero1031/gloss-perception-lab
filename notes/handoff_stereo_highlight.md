# Monocular highlight maps from stereo or render supervision (Nov 2026, paused)

## Summary

**Problem.** Every model rates matte materials on finely detailed shapes (dragon, statuette, Buddha) as glossy.

**Idea.** Use information available only at training time, as binocular vision might during development, and test on
a single image:

- **b2**: the left-right residual after warping the right view onto the left with exact correspondence. Highlights
  move with the viewpoint, shading does not.
- **b1**: the rendered specular share of each pixel (as in Prokott & Fleming 2022).

A U-Net was trained on 10k random rendered scenes for each target. Its maps were read out as cues and evaluated on
Serrano leave-one-shape-out (LOSO).

**Result.** Neither network beats the simple highlight cues. On held-out renders the maps are well calibrated. On
Serrano matte images of detailed shapes they still over-respond.

## Results (Serrano LOSO, mean of 9 shapes)

| Model | r | bal-MSE | AUC | abs. shape-mean error | mean prediction on matte images |
|---|---|---|---|---|---|
| Highlight cues | **0.639** | **0.083** | **0.834** | **1.10** | **2.81** |
| U-Net RGB | 0.625 | 0.113 | 0.811 | 1.58 | 3.23 |
| b2 residual, grey background | 0.514 | 0.096 | 0.764 | 1.37 | 3.27 |
| b1 specular, grey background | 0.415 | 0.100 | 0.733 | 1.51 | 3.54 |
| b2, scene background (earlier) | 0.468 | | | | 3.37 |

Human rating of the matte images is 1. Full output: `runs/highlight_cues/serrano_highlight_eval.txt`; earlier models in
`runs/v10k_scene_bg/` and `runs/prelim_9k/`.

- **Dragon alone**: r 0.27 (cues), 0.41 (U-Net), 0.31 (b2); mean prediction on matte images 4.1-5.4.
- **Matte vs glossy within a shape** (AUC of the mean b2 map): improved with the grey background, e.g. dragon
  0.64 -> 0.79, teapot 0.70 -> 0.87, bunny 0.74 -> 0.85.
- **For contrast, sphere only with unseen materials**: U-Net 0.90, cues 0.83-0.86, LoG responses + MLP 0.83-0.86, noise
  ceiling 0.94. The task is easy within a shape and hard across shapes.

## Diagnosis

1. **Geometry alone is not the cause.** Serrano's dragon, bunny, Buddha and sphere rendered with our own materials and
   Serrano's HDRs give low matte maps (target 0.004-0.013, prediction 0.006-0.026). Serrano's matte images give
   0.02-0.04, as high as our glossy renders (`runs/serrano_render/summary.txt`).
2. **The background mattered.** Networks trained with scene backgrounds responded to the background and to shapes that
   fill the frame. A grey background (128/255) in training and testing raised r from 0.468 to 0.514. Test images must
   be composited with the full silhouette; an eroded mask cuts off the rim.
3. **Exposure is no longer a factor.** Serrano's matte images are dark and fell outside the training exposure range
   (0.6-1.6). After widening it to 0.08-1.6, re-exposing the test images changes nothing.
4. **Human discounting does not explain it.** Only 4-10% of the matte-rated images on detailed shapes are materials rated
   3 or higher on the sphere.
5. **Where the errors are** (`runs/figures/false_highlights.png`, top 3% of object pixels per image):
   - The gloss map P fires on lit convex ridges flanked by shadowed grooves. A centre-surround filter cannot tell a
     bright shading band from a highlight.
   - b2 fires on silhouettes and thin parts (horns, ears, arm edges). The residual target is 8.6 times higher within
     2 px of the silhouette even on diffuse objects, so the network has probably learnt this artefact (not verified).
6. **Remaining suspects**: Serrano's measured BRDFs (our renders used Principled diffuse or GGX), the renderer
   (Serrano used Mitsuba), and framing.

## Related work

- **Serrano et al. 2021** feed the whole image to a VGG16 without a mask; observers were told to attend to the object.
  Their unseen-geometry test (bunny) keeps the same materials in training.
- **Prokott & Fleming 2022** use frame-filling undulating surfaces without background.
- **Morimoto et al. (bioRxiv 2025)** render 36 shapes x 36 environments x 3 views on a grey background with a fixed Ward
  roughness of 0.05 and no matte materials. One 15x15 kernel reaches r 0.64 with humans (ceiling about 0.85). The hard
  part of our setting, matte vs glossy and varying roughness, is absent there.

## Open options (time estimates)

1. Remove the silhouette band from the b2 target and retrain: about 35 min for targets, 1 h 15 min for two models and
   evaluation. Tests the contour hypothesis in diagnosis 5.
2. Match object size and framing to Serrano (about 10 min).
3. Add the map cues to the highlight-cue model (about 5 min).
4. Training data rendered with measured BRDFs (MERL). Cycles cannot use tabulated BRDFs, so use Mitsuba 3 or the fits
   from `scripts/merl_fit.py`.
5. LOSO for TinyGlossNet (30-45 min).
6. Stop here and write this up as a negative result.

## Reproduction

```powershell
# 10k scenes (Blender 4.5, GPU, about 4.5 s per scene, 250 scenes per call)
blender -b --factory-startup -P scripts/stereo_dataset.py -- --out data/stereo_ds --start 0 --count 250
.\.venv\Scripts\python.exe -m scripts.stereo_targets                       # -> data/stereo_ds_targets.npz (about 2 GB)
.\.venv\Scripts\python.exe -m scripts.train_highlight --target residual    # b2 -> runs/highlight_residual
.\.venv\Scripts\python.exe -m scripts.train_highlight --target spec        # b1 -> runs/highlight_spec
.\.venv\Scripts\python.exe -m scripts.serrano_highlight_eval               # -> runs/highlight_cues/serrano_highlight_eval.txt
blender -b --factory-startup -P scripts/serrano_shape_render.py -- --out data/serrano_render
.\.venv\Scripts\python.exe -m scripts.serrano_render_eval                  # -> runs/serrano_render/
.\.venv\Scripts\python.exe -m scripts.data_overview                        # -> runs/figures/train_scenes.png, serrano_test_images.png
.\.venv\Scripts\python.exe -m scripts.false_highlight_figure               # -> runs/figures/false_highlights.png
```

Defaults: grey background, background target 0, zoom crops 0.45-1, exposure 0.08-1.6, 128 px, U-Net base 16 / depth 4,
30 epochs (about 30 min per model). Serrano geometry and HDRs go in `data/serrano/geometry`, extracted from Serrano's
`materials_illuminations_geometries.zip`.
