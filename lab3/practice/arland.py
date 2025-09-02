import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def histogram(image):
    h, w = image.shape
    total = h * w
    hist = np.zeros(256, dtype=np.uint32)
    for i in range(h):
        for j in range(w):
            hist[image[i, j]] += 1

    pdf = np.zeros(256, dtype=np.float32)
    for i in range(256):
        pdf[i] = hist[i] / float(total)

    cdf = np.zeros(256, dtype=np.float32)
    p = 0
    for i in range(256):
        p += pdf[i]
        cdf[i] = p

    trans = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        trans[i] = int(round(255 * cdf[i]))

    out_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out_image[i, j] = trans[image[i, j]]

    return out_image, hist, pdf, cdf


def histogram_matching(source, target_pdf):
    h, w = source.shape
    total = h * w

    src_hist = np.zeros(256, dtype=np.uint32)
    for i in range(h):
        for j in range(w):
            src_hist[source[i, j]] += 1

    src_pdf = np.zeros(256, dtype=np.float32)
    for i in range(256):
        src_pdf[i] = src_hist[i] / float(total)

    src_cdf = np.zeros(256, dtype=np.float32)
    p = 0
    for i in range(256):
        p += src_pdf[i]
        src_cdf[i] = p

    target_cdf = np.zeros(256, dtype=np.float32)
    p = 0
    for i in range(256):
        p += target_pdf[i]
        target_cdf[i] = p

    trans = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        min_diff = 1.0
        index = 0
        for j in range(256):
            diff = abs(src_cdf[i] - target_cdf[j])
            if diff < min_diff:
                min_diff = diff
                index = j
        trans[i] = index

    matched_image = np.zeros_like(source, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            matched_image[i, j] = trans[source[i, j]]

    matched_hist = np.zeros(256, dtype=np.uint32)
    for i in range(h):
        for j in range(w):
            matched_hist[matched_image[i, j]] += 1

    matched_pdf = np.zeros(256, dtype=np.float32)
    for i in range(256):
        matched_pdf[i] = matched_hist[i] / float(total)

    matched_cdf = np.zeros(256, dtype=np.float32)
    p = 0
    for i in range(256):
        p += matched_pdf[i]
        matched_cdf[i] = p

    return matched_image, matched_hist, matched_pdf, matched_cdf


def erlang_pdf_loop(k, mu):
    pdf = np.zeros(256, dtype=np.float32)
    denom = (mu ** k) * math.gamma(k)
    for x in range(256):
        pdf[x] = ((x ** (k - 1)) * math.exp(-x / mu)) / denom
    total = 0
    for i in range(256):
        total += pdf[i]
    for i in range(256):
        pdf[i] /= total
    return pdf


img = cv2.imread("lena.jpg", 0)

k = float(input("Enter shape parameter k: "))
mu = float(input("Enter scale parameter µ: "))

target_pdf = erlang_pdf_loop(k, mu)

plt.figure()
plt.plot(range(256), target_pdf, color='red')
plt.title("Target Erlang PDF")
plt.show()

matched_img, matched_hist, matched_pdf, matched_cdf = histogram_matching(img, target_pdf)
in_img, in_hist, in_pdf, in_cdf = histogram(img)

cv2.imshow("Input Image", img)
cv2.imshow("Histogram Matched Image", matched_img)
cv2.waitKey(0)

plt.figure(figsize=(12, 5))
plt.subplot(2, 3, 1)
plt.bar(range(256), in_hist, color='gray')
plt.title("Input Histogram")
plt.subplot(2, 3, 2)
plt.plot(range(256), in_pdf, color='blue')
plt.title("Input PDF")
plt.subplot(2, 3, 3)
plt.plot(range(256), in_cdf, color='green')
plt.title("Input CDF")
plt.subplot(2, 3, 4)
plt.bar(range(256), matched_hist, color='gray')
plt.title("Output Histogram")
plt.subplot(2, 3, 5)
plt.plot(range(256), matched_pdf, color='blue')
plt.title("Output PDF")
plt.subplot(2, 3, 6)
plt.plot(range(256), matched_cdf, color='green')
plt.title("Output CDF")
plt.tight_layout()
plt.show()

cv2.destroyAllWindows()
