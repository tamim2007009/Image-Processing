#!/usr/bin/env python3
"""
OMR Image Display Summary

This document lists all the intermediate images that will be displayed
when running the OMR application with cv2.imshow() enabled.
"""

IMAGE_DISPLAY_SEQUENCE = [
    "1. Original Input Image - The raw image as loaded from file",
    "2. Resized Input Image - Image resized to 400x500 pixels",
    "3. Grayscaled Image - Converted to grayscale for processing",
    "4. Blurred Image - Gaussian blur applied to reduce noise",
    "5. Edge Detected Image - Canny edge detection applied",
    "6. Contours - All detected contours drawn on the image",
    "7. Corners - Corner points of detected shapes",
    "8. Answer Section - The largest detected area (answer sheet section)",
    "9. Birds Eye View (before cropping) - Answer section extracted",
    "10. Birds Eye View - Cropped lower 4/5th portion",
    "11. Thresholded Image - Binary image after thresholding",
    "12. Answers - Detected answers with correct/incorrect marking",
    "13. Grading Section - Grade area highlighted",
    "14. Grading - Final grade displayed on image"
]

def print_display_info():
    """Print information about image display sequence"""
    print("=== OMR IMAGE DISPLAY SEQUENCE ===")
    print("The following images will be displayed in order:")
    print()
    for item in IMAGE_DISPLAY_SEQUENCE:
        print(item)
    print()
    print("• Each image will be displayed in a separate window")
    print("• Press any key to proceed to the next image")
    print("• Image information (shape, type) will be printed to terminal")
    print("• Final results will be displayed in terminal with PASS/FAIL status")

if __name__ == "__main__":
    print_display_info()