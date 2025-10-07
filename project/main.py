import cv2
import numpy as np
# from openpyxl.workbook import Workbook  # Removed - no longer saving to Excel

import convolution
import helper
import gaussian_filter
import canny_edge_detection

import os

def imageShow(image, title="Image", show_image=True):
    """Display image optionally and print image info"""
    print(f"[{title}] Shape: {image.shape}, Type: {image.dtype}")
    if show_image:
        cv2.imshow(title, image)
        cv2.waitKey(0)

def display_results_terminal(student_id, score, threshold=50.0):
    """Display results in terminal with pass/fail based on threshold"""
    status = "PASS" if score >= threshold else "FAIL"
    print(f"Student ID: {student_id}")
    print(f"Score: {score:.1f}%")
    print(f"Status: {status} (Threshold: {threshold}%)")
    print("-" * 40)

id=[]
res=[]
import helper
import gaussian_filter
import canny_edge_detection

import os

def imageShow(image, title="Image", show_image=True):
    """Display image optionally and print image info"""
    print(f"[{title}] Shape: {image.shape}, Type: {image.dtype}")
    if show_image:
        cv2.imshow(title, image)
        cv2.waitKey(0)

def display_results_terminal(student_id, score, threshold=50.0):
    """Display results in terminal with pass/fail based on threshold"""
    status = "PASS" if score >= threshold else "FAIL"
    print(f"Student ID: {student_id}")
    print(f"Score: {score:.1f}%")
    print(f"Status: {status} (Threshold: {threshold}%)")
    print("-" * 40)

id=[]
res=[]
def processImages(filepath):

    # img=cv2.imread("omr.png")
    img = cv2.imread(filepath)
    filename = os.path.basename(filepath)
    print("Filename:", filename)
    filename_without_extension = filename.split('.')[0]
    print("Filename without extension:", filename_without_extension)
    id.append(filename_without_extension)

    # Show original input image before processing
    imageShow(img, "Original Input Image", show_image=True)

    width = 400
    height = 500

    questions = 5
    choices = 5
    ans = [1, 2, 0, 1, 4]

    # RESIZING
    img = cv2.resize(img, (width, height))
    imageShow(img, "Resized Input image", show_image=True)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imageShow(imgGray, "Grayscaled image", show_image=True)

    # BLURRING
    kernel = gaussian_filter.gaussian(1, 1)
    imgBlur = convolution.convolution(imgGray, kernel)
    imgBlur = convolution.normalization(imgBlur)
    imageShow(imgBlur, "Blurred image", show_image=True)

    # EDGE DETECTION
    imgCanny = canny_edge_detection.canny(imgBlur)
    imageShow(imgCanny, "Edge detected image", show_image=True)

    # FINDING ALL CONTOURS
    contours = helper.get_edge_points(imgCanny)
    # print(contours)
    # print(len(contours))
    drawCnt = img.copy()
    for i in range(len(contours)):
        # print("Contours")
        # print(contours[i])
        # contour_points = np.array(contours[i], dtype=np.int32)
        helper.manual_draw_contours(drawCnt, contours[i], (0, 255, 0), 1)
        # print("done")
    imageShow(drawCnt, "Contours", show_image=True)

    # FIND CORNERS
    corner_points = []
    for contour in contours:
        corner_list = helper.find_corners(contour)
        corner_points.append(corner_list)
    # print(corner_points)

    drawCorners = img.copy()
    for corner in corner_points:
        helper.manual_draw_contours(drawCorners, corner, (0, 255, 0), 2)
    imageShow(drawCorners, "Corners", show_image=True)

    # FIND AREA
    areas = []
    for corners in corner_points:
        rec_width1 = abs(corners[1][0] - corners[0][0])
        rec_width2 = abs(corners[3][0] - corners[2][0])
        rec_width = (rec_width2 + rec_width1) / 2
        rec_height1 = abs(corners[0][1] - corners[2][1])
        rec_height2 = abs(corners[1][1] - corners[3][1])
        rec_height = (rec_height2 + rec_height1) / 2
        area = rec_width * rec_height
        areas.append(area)
    print("Area")
    print(areas)

    sorted_area = sorted(areas, reverse=True)
    print(sorted_area)
    max_area = sorted_area[0]
    area_index = []
    for i in range(len(sorted_area)):
        for j in range(len(areas)):
            if (sorted_area[i] == areas[j]):
                area_index.append(j)
                break;
    # print(area_index)
    max_index = area_index[0]
    max_contour = img.copy()
    # 0 -> ans, 2 -> grade, 4 -> name
    helper.manual_draw_contours(max_contour, contours[area_index[0]], (0, 255, 0), 1)

    imageShow(max_contour, "Answer section", show_image=True)

    ans_corner_points = corner_points[max_index]
    bl_x = ans_corner_points[2][0] - 10
    bl_y = ans_corner_points[2][1] - 10
    tr_x = ans_corner_points[1][0] + 5
    tr_y = ans_corner_points[1][1] + 20
    # anss = np.ones((row_end-row_start+1,col_end-col_start+1))
    wd = tr_x - bl_x
    ht = tr_y - bl_y
    x, y, w, h = bl_x, bl_y, wd, ht  # Example: x, y, width, height of the ROI
    roi = imgGray[y:y + h, x:x + w]
    ans_new_image = np.zeros_like(roi)  # Create a black image with the same size as roi
    ans_new_image[:, :] = roi
    imageShow(ans_new_image, "Birds eye view (before cropping)", show_image=True)

    height, width = ans_new_image.shape[:2]
    # Define the crop region for the lower 4/5th portion
    crop_height = int(height * 4 / 5)  # Calculate the crop height

    # Crop the lower 4/5th portion of the image
    lower_portion = ans_new_image[-crop_height:, :]
    img_padded = cv2.copyMakeBorder(lower_portion, 1, 1, 3, 0, cv2.BORDER_CONSTANT)
    imageShow(img_padded, "Birds eye view", show_image=True)
    # print(img_padded.shape)

    imgThres = np.zeros_like(img_padded)
    imgThres = helper.thresholdImage(img_padded, imgThres, 170)
    imageShow(imgThres, "Thresholded image", show_image=True)

    boxes = helper.splitBoxes(imgThres)
    # for i in range(25):
    #     cv2.imshow("Test box", boxes[i])
    #     cv2.waitKey(0)

    # GETTING NON ZERO PIXEL VALUES OF EACH BOX
    myPixelVal = np.zeros((questions, choices))
    countC = 0
    countR = 0

    for image in boxes:
        totalPixels = helper.countNonZeroPixel(image)
        # print("pixel count")
        # print(totalPixels)
        # cv2.imshow("Test box", image)
        # cv2.waitKey(0)

        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if (countC == choices):
            countR += 1
            countC = 0
    # print("Count pixel value")
    # print(myPixelVal)

    # FINDING INDEXES OF THE MARKINGS
    myIndex = []
    for x in range(0, questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        # print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    # print(myIndex)

    # GRADING
    gradings = []
    for x in range(0, questions):
        if (ans[x] == myIndex[x]):
            gradings.append(1)
        else:
            gradings.append(0)
    score = (sum(gradings) / questions) * 100
    res.append(score)
    # print(score)

    # DISPLAYING ANSWERS
    imgResult = img_padded.copy()
    imgResult = helper.showAnswers(imgResult, myIndex, questions, ans, choices, gradings)
    imageShow(imgResult, "Answers", show_image=True)

    # DISPLAY GRADING
    grade_index = area_index[2]
    grade_contour = img.copy()
    # 0 -> ans, 2 -> grade, 4 -> name
    # helper.manual_draw_contours(grade_contour, contours[grade_index], (0, 255, 0), 1)
    # cv2.imshow("Grade contour",grade_contour)
    # cv2.waitKey(0)

    grade_corner_points = corner_points[grade_index]
    # print("grade corner points")
    # print(grade_corner_points)
    grade_bl_x = grade_corner_points[2][0] - 50
    grade_bl_y = grade_corner_points[2][1] - 100
    grade_tr_x = grade_corner_points[1][0] + 20
    grade_tr_y = grade_corner_points[1][1] + 20
    # anss = np.ones((row_end-row_start+1,col_end-col_start+1))
    grade_wd = grade_tr_x - grade_bl_x
    grade_ht = grade_tr_y - grade_bl_y
    x, y, w, h = grade_bl_x, grade_bl_y, grade_wd, grade_ht  # Example: x, y, width, height of the ROI
    grade_roi = imgGray[y:y + h, x:x + w]
    grade_new_image = np.zeros_like(grade_roi)  # Create a black image with the same size as roi
    grade_new_image[:, :] = grade_roi
    helper.manual_draw_contours(grade_contour, contours[grade_index], (0, 255, 0), 1)
    imageShow(grade_contour, "Grading section", show_image=True)

    imgGrading = grade_contour.copy()
    cv2.putText(imgGrading, str(int(score)) + "%", (240, 390), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
    imageShow(imgGrading, "Grading", show_image=True)

    # Display results in terminal with 50% threshold
    display_results_terminal(filename_without_extension, score, threshold=50.0)



    #
    # imgBlank = np.zeros_like(img)
    # imageArray = [[img,imgGray,imgBlur,imgCanny],
    #               [drawCnt,max_contour,img_padded,imgThres],
    #               [imgResult,grade_contour,imgGrading,imgBlank]]
    # lables = [["Original","Gray","Blur","Canny"],
    #           ["Contours","Ans section","Birds eye","Threshold"],
    #           ["Result","Grade section","Grading","Blank"]]
    #
    # imgStacked = helper.stackImages(imageArray,0.5,lables)
    # #
    # # cv2.imshow("Final Result",imgFinal)
    # cv2.imshow("Stacked Images",imgStacked)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_to_excel(id_array, score_array):
    """
    Excel saving function - currently disabled
    Uncomment the imports and code below if you want to enable Excel saving
    """
    # from openpyxl.workbook import Workbook
    # wb = Workbook()
    # ws = wb.active
    # ws.title = "ID Scores"
    # 
    # # Write headers
    # ws['A1'] = "ID"
    # ws['B1'] = "Score"
    # 
    # # Write data
    # for idx, (id_val, score_val) in enumerate(zip(id_array, score_array), start=2):
    #     ws[f'A{idx}'] = id_val
    #     ws[f'B{idx}'] = score_val
    # 
    # # Save the workbook
    # excel_filename = "result.xlsx"
    # wb.save(excel_filename)
    # print(f"Excel file saved to: {excel_filename}")
    
    print("Excel saving is disabled. Results are displayed in terminal only.")

def main():
    global id, res
    
    print("=== Optical Mark Recognition System ===")
    print("Enter image file paths (one per line). Type 'done' when finished:")
    
    image_paths = []
    while True:
        path = input("Image path: ").strip()
        if path.lower() == 'done':
            break
        if path and os.path.exists(path):
            image_paths.append(path)
            print(f"Added: {path}")
        else:
            print("File not found. Please enter a valid path.")
    
    if not image_paths:
        print("No valid images provided. Exiting.")
        return
    
    print(f"\nProcessing {len(image_paths)} image(s)...")
    
    for file_path in image_paths:
        print(f"\nProcessing: {file_path}")
        try:
            processImages(file_path)
            print("✓ Processing completed")
        except Exception as e:
            print(f"✗ Error processing {file_path}: {str(e)}")
    
    # Save results to Excel
    if id and res:
        # save_to_excel(id, res)  # Removed Excel saving
        print(f"\n{'='*50}")
        print("FINAL SUMMARY")
        print(f"{'='*50}")
        pass_count = sum(1 for score in res if score >= 50.0)
        fail_count = len(res) - pass_count
        print(f"Total Students: {len(res)}")
        print(f"Passed (≥50%): {pass_count}")
        print(f"Failed (<50%): {fail_count}")
        print(f"Pass Rate: {(pass_count/len(res)*100):.1f}%")
        print(f"{'='*50}")
        for i, (student_id, score) in enumerate(zip(id, res)):
            status = "PASS" if score >= 50.0 else "FAIL"
            print(f"{student_id}: {score:.1f}% - {status}")
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()