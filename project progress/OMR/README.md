## OMR

A straightforward app to scan and read marked answers and evaluate result.

### Project Features

- Multiple image upload.
- Gray scale conversion.
- Smoothing.
- Edge detection.
- Contour detection.
- Perspective transformation.
- Result sheet.

## Processing Pipeline

The core workflow implemented in `processImages(filepath)` follows these steps:

- 1. Load & Resize Image
- 2. Grayscale Conversion
- 3. Image Preprocessing
  - Gaussian Blur (cv2.GaussianBlur)
  - Canny Edge Detection (cv2.Canny)
- 4. Contour Detection & Analysis
  - helper.get_edge_points()
  - helper.find_corners()
  - Area calculation
- 5. ROI Extraction
  - Answer section (largest area)
  - Grade section (3rd largest)
- 6. Answer Detection
  - helper.thresholdImage()
  - helper.splitBoxes()
  - helper.countNonZeroPixel()
- 7. Grading Logic
  - Compare with answer key
- 8. Result Visualization
  - helper.showAnswers()

### Project Snapshots

<table>
  <tr>
    <td>
      <img src="https://github.com/ZakariaHossain56/OMR/raw/master/snapshots/gui.png" alt="GUI" width="300">
      <p style="text-align: center;">GUI</p>
    </td>
    <td>
      <img src="https://github.com/ZakariaHossain56/OMR/raw/master/snapshots/input.png" alt="OMR Sheet" width="300">
      <p style="text-align: center;">OMR Sheet</p>
    </td>
    <td>
      <img src="https://github.com/ZakariaHossain56/OMR/raw/master/snapshots/gray.png" alt="Grayscale" width="300">
      <p style="text-align: center;">Grayscale</p>
    </td>
  </tr>

  <tr>
    <td>
      <img src="https://github.com/ZakariaHossain56/OMR/raw/master/snapshots/blur.png" alt="Smoothing" width="300">
      <p style="text-align: center;">Smoothing</p>
    </td>
    <td>
      <img src="https://github.com/ZakariaHossain56/OMR/raw/master/snapshots/canny.png" alt="Edge detection" width="300">
      <p style="text-align: center;">Edge detection</p>
    </td>
    <td>
      <img src="https://github.com/ZakariaHossain56/OMR/raw/master/snapshots/contour.png" alt="Contour detection" width="300">
      <p style="text-align: center;">Contour detection</p>
    </td>
  </tr>

<tr>
    <td>
      <img src="https://github.com/ZakariaHossain56/OMR/raw/master/snapshots/show_ans.png" alt="Birds eye view" width="300">
      <p style="text-align: center;">Birds eye view</p>
    </td>
    <td>
      <img src="https://github.com/ZakariaHossain56/OMR/raw/master/snapshots/show_grade.png" alt="Grading" width="300">
      <p style="text-align: center;">Grading</p>
    </td>
    <td>
      <img src="https://github.com/ZakariaHossain56/OMR/raw/master/snapshots/result.png" alt="Result sheet" width="300">
      <p style="text-align: center;">Result sheet</p>
    </td>
  </tr>

</table>

### Directory Structure

```
📂OMR
 |───main.py
 │───canny_edge_detection.py
 │───convolution.py
 |───derivative.py
 |───gaussian_filter.py
 |───grayscale_convolution.py
 |───sobel_filter.py
 └───helper.py

```

### Libraries

[OpenCV](https://pypi.org/project/opencv-python/) <br>
[NumPy](https://pub.dev/packages/provider) <br>
[openpyxl](https://pub.dev/packages/audioplayers) <br>
[Tkinter](https://pub.dev/packages/path_provider) <br>
[Pillow](https://pub.dev/packages/permission_handler) <br>
