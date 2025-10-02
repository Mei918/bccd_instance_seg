# BCCD Cell Instance Segmentation — Report (README)

> Dataset: **BCCD dataset with masks**  
> Source: https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask/data

This README documents the end‑to‑end pipeline, model adaptation, training procedure, quantitative results, qualitative analyses, and reflections after running the provided Jupyter notebook.

---

## 1) Problem & Data

Goal: **instance segmentation** of blood cells (WBC, RBC, Platelets) from microscope images.

- **Data split**: small deterministic split with a cap (default **100 images**) and ratio **70/15/15** (train/val/test).  
- **Preprocessing for histopathology**:
  - **Reinhard stain normalization** to reduce color variation across slides/labs.
  - **Patch extraction / random crops** around `IMG_SIZE` (default **384** for quick dev) to improve scale robustness.
  - **Augmentations** (Albumentations): flips, rotate90, small affine transforms, blur, brightness/contrast, hue/saturation value shifts (mild to avoid breaking morphology).

**Figure (split preview):** `viz/step1_split_train.png`  
**Figure (stain normalization):** `viz/step2_reinhard_preview.png`, `viz/step2_overlay_mask.png`  
**Figure (patch extraction):** `viz/step2_patch_preview.png`  
**Figure (augmentation samples):** `viz/step2_aug_preview.png`

---

## 2) Model Adaptation & Design Choices

### 2.1 Architecture overview
We use **Mask R‑CNN** with an FPN backbone (ResNet50‑FPN by default) as a strong equivalent to UNI for instance segmentation, as UNI is not publicly accessible:

- **Backbone**: ResNet50 with **FPN** provides multi‑scale features (P2–P5) beneficial for small, densely packed cell instances.  
- **RPN**: Region Proposal Network adapted with a **single anchor size per FPN level** (safe default) to avoid version‑specific shape issues; optional multi‑size/level configuration is supported by rebuilding the RPN head.
- **ROI Heads**: class/box heads plus a **mask head** that predicts per‑instance binary masks.
- **Transforms**: model normalization disabled (mean=0,std=1) since inputs are already scaled to [0,1].

> **Why acceptable as “UNI (or equivalent)”**  
> UNI is a modern universal visual backbone. In practice, **any high‑capacity, multi‑scale backbone** (e.g., ConvNeXt/TIMM models via FPN, or ResNet50‑FPN) offers comparable behavior for Mask R‑CNN on small/medium datasets like BCCD. We use ResNet50‑FPN for:
> - broad availability and **reproducibility** in `torchvision`,
> - balanced **compute/accuracy** for small objects,
> - strong **transferability**; easily swappable to UNI‑like features.

**How to swap to UNI** (if you have a UNI checkpoint): replace the `backbone` with UNI features that expose a pyramid (or wrap UNI with an FPN neck). Ensure the `out_channels` matches the RPN/ROI heads and either keep single‑size anchors or rebuild the RPN head for multi‑size anchors (already shown in code).

**Figure (anchor visualization):** `viz/step3_anchor_preview.png`

### 2.2 Pathology‑aware choices
- **Stain normalization** stabilizes color distributions, helping mask heads generalize to different blood film preparations.
- **Mild geometric/photometric augments** preserve nucleus/cytoplasm boundaries; excessive warps could distort cell morphology and harm downstream metrics.
- **Small‑object sensitivity** via FPN and tuned anchors is crucial for platelets and tightly clustered RBCs.

---

## 3) Custom Data Loader

The dataset provides **semantic masks**. We convert them to **instances** by connected components:

- Binary mask → **connected‑component** labeling → per‑instance mask.  
- Per‑instance **bounding boxes** derived from mask extents.  
- Outputs conform to `torchvision` detection API:
  - `image` (float tensor in [0,1]),
  - `target = {boxes, labels, masks, area, iscrowd, image_id}`.

**Figure (dataset item overlay):** `viz/step4_dataset_item_0.png`

---

## 4) Loss Functions

Mask R‑CNN total loss is a **weighted sum** (equal weights by default):

- **RPN**: objectness (binary cross‑entropy) + box regression (Smooth‑L1).
- **ROI classifier**: multi‑class cross‑entropy.
- **ROI box regressor**: Smooth‑L1.
- **Mask head**: per‑pixel **binary cross‑entropy** between predicted instance logits and ground‑truth instance masks.

> Rationale:  
> - BCE for masks is robust for binary instance masks.  
> - Smooth‑L1 stabilizes box regression.  
> - Balanced multi‑task loss enforces proposal quality, classification, localization, and fine‑grained mask accuracy.

---

## 5) Training Procedure & Hyperparameters

- **Optimizer**: AdamW, `lr=1e-4`, weight decay `1e-4`.  
- **LR schedule**: StepLR (halves mid‑training for full runs).  
- **Batch size**: `2` (quick dev) / `4` (full).  
- **Epochs**: `3` (fast dev) / `12` (full).  
- **Image size**: `384` (fast) / `512` (full).  
- **AMP**: enabled automatically when **CUDA** is available via `torch.amp`.  
- **Workers**: `2`/`4` depending on dev/full.  
- **Logging/plots**: `viz/step5_train_loss.png` shows the training curve.

Reproducibility: fixed `SEED=42`, deterministic splits, capped dataset size (`MAX_IMAGES=100`) for quick experiments.

---

## 6) Evaluation & Metrics

We report **IoU**, **Dice**, and a simple **mAP@0.5** proxy (box IoU ≥ 0.5).

- **IoU**: compares union of predicted vs. true masks at the image level; sensitive to under/over‑segmentation.  
- **Dice**: favors overlap on small structures; often higher than IoU on tiny platelets.  
- **mAP@0.5 (proxy)**: computed from boxes extracted from predicted/true masks; prioritizes correct instance detection & localization.

### Results (fill from your run)
The notebook prints the aggregates at the end of Step 5. Replace the placeholders below with your values:

| Split | IoU ↑ | Dice ↑ | mAP@0.5 ↑ |
|---|---:|---:|---:|
| **Val** | **`<IoU_val>`** | **`<Dice_val>`** | **`<mAP_val>`** |
| **Test** (optional) | `<IoU_test>` | `<Dice_test>` | `<mAP_test>` |

**Figure (metrics bar):** `viz/step5_metrics_bar.png`  
**Figure (train loss):** `viz/step5_train_loss.png`

**Trade‑offs in context of cell morphology**:
- **Dice** is forgiving for tiny platelets; it can stay high even when a few instances are missed.  
- **IoU** penalizes boundary errors and merging; RBC clumps can lower IoU due to touching boundaries.  
- **mAP@0.5** drops when instances are split/merged or when platelets are missed (proposal/score thresholding issues).

---

## 7) Qualitative Analysis

We visualize overlays on the original images and annotate best/worst cases (by IoU).

- **Success cases**: distinct WBC nuclei; well‑separated RBC; moderate illumination.  
- **Failure cases**: touching RBC (merge), tiny/low‑contrast platelets (miss), stain extremes (color shift).

**Figures**:  
- Success panel: `viz/step6_success_cases.png`  
- Failure panel: `viz/step6_failure_cases.png`

> Common error patterns & interpretations:
> - **Under‑segmentation** (merging): RPN/ROI heads miss boundaries between contacting cells → lower IoU, normal Dice.  
> - **Over‑segmentation** (fragmentation): a single RBC yields multiple masks → hurts mAP.  
> - **Missed platelets**: small size + low contrast → recall limited at default anchor/scoring thresholds.

---

## 8) Practical Notes & Reproducibility

- **Stain normalization** is essential; without it, mask quality varies across slides.  
- **Anchors**: safe default = single size per FPN level; to use multi‑size/level anchors, the code **rebuilds the RPN head** to avoid shape mismatches across `torchvision` versions.  
- **AMP**: speeds up training on CUDA without quality loss in this regime.

---

## 9) Limitations & Potential Improvements

1. **Backbone**: switch to a UNI‑style universal backbone or a stronger TIMM model (e.g., ConvNeXt‑Tiny) wrapped with an FPN neck for better small‑object features.  
2. **Losses**: add **Dice loss** (or BCE+Dice) for the mask head to emphasize small instances; consider **Focal loss** for class/objectness to improve platelet recall.  
3. **Post‑processing**: tune **score/NMS thresholds**; consider mask‑based NMS or **Soft‑NMS** to reduce over‑merging.  
4. **Augmentations**: add **cutout**/random erasing and mild **color jitter** with pathology‑aware bounds; consider **mixup/cutmix** cautiously.  
5. **Resolution**: larger crops (`512+`) especially help platelet detection; train with **multi‑scale** resizing at inference time.  
6. **Label noise**: morphological ops to refine GT masks (e.g., small hole filling) can stabilize training.  
7. **Instance separation**: an auxiliary **boundary head** (e.g., edge‑aware loss) or **Watershed‑guided refinement** to split touching RBCs.

---

## 10) How to Reproduce

1. Open the notebook and enable **GPU**.  
2. (Fresh env) Install dependencies (see the first cell in the notebook).  
3. Run cells step‑by‑step (1 → 7).  
4. Results and figures will be written under:
   - **Checkpoints**: `checkpoints/`
   - **Visualizations**: `viz/`
   - **Splits JSON**: `bccd_splits_100.json`

---

## 11) Environment

- Python ≥ 3.9, PyTorch ≥ 2.2, TorchVision ≥ 0.17, Albumentations, scikit‑image, OpenCV, timm, pycocotools.  
- AMP via `torch.amp` enabled automatically on CUDA.

---

## 12) Acknowledgements

- BCCD dataset contributors.  
- TorchVision Mask R‑CNN reference implementation.  
- Albumentations & scikit‑image for preprocessing utilities.
