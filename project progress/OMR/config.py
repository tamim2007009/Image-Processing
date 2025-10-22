"""
Configuration file for OMR System
Modify these settings to customize the OMR sheet processing
"""

# ==================== IMAGE PROCESSING SETTINGS ====================

# Target image dimensions after resizing
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 500

# Gaussian blur parameters
BLUR_KERNEL_SIZE = (5, 5)  # Must be odd numbers
BLUR_SIGMA = 1

# Canny edge detection thresholds
CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150

# Thresholding value for answer detection
THRESHOLD_VALUE = 170


# ==================== ANSWER SHEET SETTINGS ====================

# Number of questions on the answer sheet
NUM_QUESTIONS = 5

# Number of choices per question (e.g., 5 for A, B, C, D, E)
NUM_CHOICES = 5

# Correct answer key (0=A, 1=B, 2=C, 3=D, 4=E)
# Modify this array according to your answer key
ANSWER_KEY = [0, 0, 0, 0, 0]  # All answers are 'A'


# ==================== ROI EXTRACTION SETTINGS ====================

# Corner adjustment for answer section extraction
# Format: (left_offset, top_offset, right_offset, bottom_offset)
ANSWER_CORNER_OFFSET = {
    'bottom_left_x': -10,   # Expand left
    'bottom_left_y': -10,   # Expand bottom
    'top_right_x': 5,       # Expand right
    'top_right_y': 20       # Expand top
}

# Crop ratio for removing header (keeps bottom portion)
# 4/5 means keep the lower 80% of the image
CROP_RATIO = 4 / 5

# Border padding for answer section
# Format: (top, bottom, left, right)
BORDER_PADDING = {
    'top': 1,
    'bottom': 1,
    'left': 3,
    'right': 0
}


# ==================== GRADING SETTINGS ====================

# Passing percentage (40% = 2 out of 5 correct)
PASS_MARK = 40.0

# Grade boundaries
GRADE_BOUNDARIES = {
    'A': 80,  # 80-100%
    'B': 60,  # 60-79%
    'C': 40,  # 40-59%
    'F': 0    # 0-39%
}


# ==================== DISPLAY SETTINGS ====================

# Show intermediate processing steps (set to False to skip cv2.imshow and cv2.waitKey)
SHOW_INTERMEDIATE_STEPS = True

# Grade section corner adjustment (for displaying grade on image)
GRADE_CORNER_OFFSET = {
    'bottom_left_x': -50,
    'bottom_left_y': -100,
    'top_right_x': 20,
    'top_right_y': 20
}

# Grade text display position and style
GRADE_TEXT_POSITION = (240, 390)
GRADE_TEXT_FONT = 1  # cv2.FONT_HERSHEY_COMPLEX = 1
GRADE_TEXT_SCALE = 1
GRADE_TEXT_COLOR = (0, 255, 0)  # Green (BGR format)
GRADE_TEXT_THICKNESS = 3


# ==================== GUI SETTINGS ====================

# GUI window dimensions
GUI_WIDTH = 600
GUI_HEIGHT = 400

# GUI color scheme
GUI_BG_COLOR = "#1E8F91"
GUI_BUTTON_COLOR = "#4CAF50"
GUI_TEXT_COLOR = "#ffffff"

# Thumbnail size for uploaded images
THUMBNAIL_SIZE = (100, 100)


# ==================== VALIDATION ====================

def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    if len(ANSWER_KEY) != NUM_QUESTIONS:
        errors.append(f"ANSWER_KEY length ({len(ANSWER_KEY)}) doesn't match NUM_QUESTIONS ({NUM_QUESTIONS})")
    
    for idx, ans in enumerate(ANSWER_KEY):
        if not (0 <= ans < NUM_CHOICES):
            errors.append(f"ANSWER_KEY[{idx}] = {ans} is invalid. Must be between 0 and {NUM_CHOICES-1}")
    
    if BLUR_KERNEL_SIZE[0] % 2 == 0 or BLUR_KERNEL_SIZE[1] % 2 == 0:
        errors.append(f"BLUR_KERNEL_SIZE must have odd dimensions, got {BLUR_KERNEL_SIZE}")
    
    if not (0 < CROP_RATIO <= 1):
        errors.append(f"CROP_RATIO must be between 0 and 1, got {CROP_RATIO}")
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print("✅ Configuration validated successfully!")
    return True


# Run validation when imported
if __name__ != "__main__":
    validate_config()
