from collections import deque

import cv2
import numpy as np
import config

def splitBoxes(img):
    """Split image into grid based on questions and choices - optimized with list comprehension"""
    rows = np.vsplit(img, config.NUM_QUESTIONS)
    boxes = [box for r in rows for box in np.hsplit(r, config.NUM_CHOICES)]
    return boxes

def showAnswers(img,myIndex,questions,answers,choices,grading):
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    secW = int(img.shape[1]/choices)  # Width divided by number of choices (columns)
    secH = int(img.shape[0]/questions)  # Height divided by number of questions (rows)

    for x in range(0,questions):

        myAns = myIndex[x]
        cX = (myAns*secW) + secW//2
        cY = (x*secH) + secH//2

        if(grading[x] == 1):
            myColor = (0,255,0)
        else:
            myColor = (0,0,255)
            correctAns = answers[x]
            cv2.circle(img, ((correctAns*secW)+secW//2, (x*secH)+secH//2), 10, (0,255,0), cv2.FILLED)

        cv2.circle(img,(cX,cY),10,myColor,cv2.FILLED)
    return img

def countNonZeroPixel(img):
    """Count non-zero pixels - optimized with NumPy"""
    # Use NumPy's built-in function - much faster than nested loops
    return np.count_nonzero(img)

def manual_draw_contours(img, contours, color, thickness):
    """
    Manually draw contours on an image.
    """
    for contour in contours:
        y=contour[1] + 10
        x=contour[0] + 10
        cv2.circle(img, (y, x), radius=1, color=color, thickness=thickness)

def find_corners(contour):
    """
    Find the bounding box corners for a custom contour - optimized.
    Custom contour format: list of (x, y) tuples
    """
    # Convert to numpy array for vectorized operations
    contour_array = np.array(contour)
    
    # Use numpy min/max for faster computation
    min_x = int(contour_array[:, 0].min())
    max_x = int(contour_array[:, 0].max())
    min_y = int(contour_array[:, 1].min())
    max_y = int(contour_array[:, 1].max())
    
    # Return 4 corners: top_left, top_right, bottom_left, bottom_right
    return [
        (min_x, max_y),  # top_left
        (max_x, max_y),  # top_right
        (min_x, min_y),  # bottom_left
        (max_x, min_y)   # bottom_right
    ]

def dfs(image, sp_x, sp_y, to_replace, replace_with):
    """DFS traversal - optimized with set for visited tracking"""
    height, width = image.shape
    parent_map = {}
    length = 0
    last = None

    stack = [(sp_x, sp_y, 0)]
    parent_map[(sp_x, sp_y)] = None

    # Pre-define directions for reuse
    directions = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    while stack:
        x, y, it = stack.pop()
        if image[x, y] != to_replace:
            continue

        image[x, y] = replace_with

        it += 1
        if it > length:
            length = it
            last = (x, y)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and image[nx, ny] == to_replace:
                if (nx, ny) not in parent_map:  # Check if not already visited
                    parent_map[(nx, ny)] = (x, y)
                    stack.append((nx, ny, it))

    # Build path from last to start
    points = []
    current = last
    while current is not None:
        points.append(current)
        current = parent_map[current]
    points.reverse()
    return points




def get_edge_points(image):
    """Extract edge contours - optimized with set for visited tracking"""
    image = image.copy()
    height, width = image.shape
    pad = 10
    image = image[pad:height - pad, pad:width - pad]
    height, width = image.shape
    contours = []

    visited = set()  # Use set instead of dict for faster lookup
    directions = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    def bfs(sx, sy):
        to_it = (sx, sy)

        while to_it is not None:
            queue = deque([to_it])
            to_it = None

            while queue:
                x, y = queue.popleft()
                if (x, y) in visited:
                    continue

                image[x, y] = 60
                visited.add((x, y))

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= height or ny < 0 or ny >= width or image[nx, ny] == 60:
                        continue

                    if image[nx, ny] == 255:
                        to_it = (nx, ny)
                        queue.clear()
                        break
                    if (nx, ny) not in visited:
                        queue.append((nx, ny))

            if to_it is None:
                break

            points = dfs(image, to_it[0], to_it[1], to_replace=255, replace_with=120)
            last_pt = points[-1]

            points = dfs(image, last_pt[0], last_pt[1], to_replace=120, replace_with=60)
            if len(points) > 20:
                contours.append(points)

            to_it = points[-1]

    for x in range(height):
        for y in range(width):
            if (x, y) not in visited:
                bfs(x, y)
    
    return contours

def thresholdImage(img, thres, threshold):
    thres = np.where(img < threshold, 255, 0).astype(np.uint8)
    return thres