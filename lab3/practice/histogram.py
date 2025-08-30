import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---- Manual Histogram Equalization Step by Step ----
def histogram(image):
    h, w = image.shape
    total = h * w
    print(f"Image size: {h} x {w}, Total pixels: {total}")

   
    hist = np.zeros(256, dtype=np.int32)
    for i in range(h):
        for j in range(w):
            pixel_val = image[i, j]
            hist[pixel_val] += 1
    print("\nHistogram (first 30 values):")
    print(hist[:30])   # print only first 30 for brevity

    # Plot histogram
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(hist)
    plt.title("Histogram")

    # ----------------
    # Step 2: PDF
    # ----------------
    pdf = np.zeros(256, dtype=np.float32)
    for i in range(256):
        pdf[i] = hist[i] / float(total)
    print("\nPDF (first 30 values):")
    print(pdf[:30])

    plt.subplot(1,3,2)
    plt.plot(pdf)
    plt.title("PDF")

    # ----------------
    # Step 3: CDF
    # ----------------
    cdf = np.zeros(256, dtype=np.float32)
    cumulative = 0.0
    for i in range(256):
        cumulative += pdf[i]
        cdf[i] = cumulative
    print("\nCDF (first 30 values):")
    print(cdf[:30])

    plt.subplot(1,3,3)
    plt.plot(cdf)
    plt.title("CDF")
    plt.show()

    # ----------------
    # Step 4: Mapping
    # ----------------
    in_map = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        in_map[i] = int(round(cdf[i] * 255))
    print("\nMapping table (first 30 values):")
    print(in_map[:30])

    # ----------------
    # Step 5: Apply Mapping
    # ----------------
    out = np.zeros_like(image, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i, j] = in_map[image[i, j]]

    return hist, pdf, cdf, in_map, out


# -------- Main Program --------
image = cv2.imread(r"C:\Users\USER\Desktop\image\Image-Processing\lab3\practice\histogram.jpg", cv2.IMREAD_GRAYSCALE)

hist, pdf, cdf, in_map, equalized_img = histogram(image)

# Show original vs equalized images
cv2.imshow("Original Image", image)
cv2.imshow("Equalized Image (Manual)", equalized_img)
cv2.imshow("Equalized Image (OpenCV)", cv2.equalizeHist(image))
cv2.waitKey(0)
cv2.destroyAllWindows()
