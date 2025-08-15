# Simple Supernova Detection for Seestar S50 smart telescope

A Python-based pipeline for detecting transient astronomical events, such as supernovae, from stacked FITS images using Seestar S50. The program aligns images, converts RGB channels to luminance, and performs difference imaging to highlight new or bright objects.

---

## Features

- **FITS Image Handling** – Supports standard FITS images with WCS metadata.
- **Image Alignment** – Aligns images using either array-based registration or WCS-based reprojection.
- **Luminance Conversion** – Converts RGB stacks to single-channel luminance using perceptual or sensor-specific weights.
- **Difference Imaging** – Highlights new objects by subtracting a reference image from a science image.
- **Robust Thresholding** – Automatically filters noise using σ-based thresholds and removes tiny spurious detections.
- **Batch Processing Friendly** – Designed to handle large sets of images without crashing the pipeline.

---

## Installation

```bash
pip install numpy astropy reproject astroalign scipy
```

---

## General work-flow

### 1. Prepare Your Images

- `REFERENCE_IMAGE`: A FITS file of the static sky (reference).
- `ANALYSIS_IMAGE`: A FITS file of the target sky, possibly containing new objects.
- (OR call directly "detect(reference img name, analyze img name)" in your pipeline)

### 2. Convert to Luminance (Optional Sensor-Specific)

```python
luminance_ref = to_luminance(ref_data)
luminance_sci = to_luminance_s50_lp(sci_data)
```

### 3. Align Images

- **Array-based alignment**:

```python
aligned_sci = align_with_astro_align(luminance_ref, luminance_sci)
```

- **WCS-based alignment**:

```python
aligned_sci = align_with_wcs(ref_path, sci_path)
```

### 4. Find Differences

```python
find_difference_arrays(luminance_ref, aligned_sci)
```

- Outputs a FITS file (`DIF_IMG`) containing only pixels above the chosen σ-threshold.

---

## Parameters
- `REFERENCE_IMAGE`: Image used as reference.
- `ANALYSIS_IMAGE`: Image used for detection.
- `NSIGMA`: Number of standard deviations above background noise to consider significant. (EDIT AS NEEDED)
- `DIF_IMG`: Output path for the difference FITS image.
- `USE_WSC`: If should align images baseed on WSC data in header or rely on astro_align 
- `USE_SEESTAR_LUMINANCE`: If should use standard perceptual weights from the sRGB or Seestar's specific

---

## Logging

All processing steps are logged via the `logging` module.

---

## Notes

- Luminance weights can be adjusted for your camera or filters to improve sensitivity.
- Tiny detections (<3 pixels) are filtered out to reduce false positives from noise.
- WCS-based alignment requires that FITS headers contain valid celestial WCS information.

---
## Sample
<img width="1633" height="601" alt="" src="https://github.com/user-attachments/assets/9b8bbb4f-e087-4036-94d9-c47f6d31d573" /><br>
-> Artifically added a new star only in red channel of the image<br>
<br>
<img width="504" height="379" alt="" src="https://github.com/user-attachments/assets/3c96dbfd-6064-48dd-bacf-70eb920a18f1" /><br>
-> Detected difference

---

## License

MIT License – free for research and personal use.

