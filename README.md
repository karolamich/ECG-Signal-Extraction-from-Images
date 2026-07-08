# ECG Signal Extraction from Images

> **Note:** This project was developed as a course project for the "Computer Vision Systems" (Systemy Wizyjne) class.

## Introduction
This project implements a computer vision algorithm aimed at the automatic extraction of digital values for 12 standard ECG channels from medical software screenshots and photos/scans of physical printouts. 

The output of the script is a CSV file containing signal values normalized to the isoelectric line and interpolated to a constant time step of 1 ms (sampling frequency of 1000 Hz).

The algorithm operates in two modes:
* **Basic version** - designed for processing digital screenshots.
* **Extended version** - optimized for processing photos and scans of ECG printouts (accounting for distortions, noise, and rotation).

## Main Features
**Basic version:**
* Time scaling based on a reference image (200 ms interval).
* Image segmentation and isolation of the chart grid using morphological operations and contour detection.
* Baseline (isoelectric line) detection for 12 channels using the Hough transform.
* Raw data extraction through local tracking of the plotted signal continuity.
* Signal amplitude normalization and linear approximation (interpolation) of missing readings to a 1 ms step.

**Extended version:**
* Perspective and image rotation correction using affine transformation.
* Automatic separation of signal panels (left and right lead panels).
* Reduction of background noise, shadows, and paper texture via background estimation using a median filter (41x41 pixel window) and hard binarization.
* Automatic time scale detection using the autocorrelation operation on the background millimeter grid.
* Medical feature extraction: automatic calculation of heart rate (BPM) via peak detection (R-waves) and RR intervals.

## Technologies Used / Requirements
* **Python**
* **OpenCV (`cv2`)** - segmentation, Hough transform, filters, morphological operations, affine transformations.
* **NumPy** - vector and matrix operations, autocorrelation.
* **SciPy** - interpolation (`scipy.interpolate`), peak detection, and signal processing (`scipy.signal`).

## File Structure
* `main.py` – main script executing the program.
* `basic_version.py` – script implementation for screenshots.
* `extended_version.py` – script implementation for photos and scans.

## Installation and Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```
Usage
Basic execution of the program for screenshots (basic version):

```bash
python main.py --input_dir /path/to/test_images --output_dir /path/to/save_csvs
```
To run the processing for photos and printout scans (extended version), add the --mode scan flag:


```bash
python main.py --input_dir /path/to/test_images --output_dir /path/to/save_csvs --mode scan
```
