import cv2
import numpy as np
from collections import deque

def splitBoxes(img):
    img = img.copy()
    rows = np.vsplit(img, 5)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 5)
        for box in cols:
            boxes.append(box)
    return boxes

def manual_draw_contours(img, contours, color, thickness):
    for contour in contours:
        y = contour[1] + 10
        x = contour[0] + 10
        cv2.circle(img, (y, x), radius=1, color=color, thickness=thickness)

def find_corners(contour):
    corners = []
    max_x = 0
    min_x = 1e6
    max_y = 0
    min_y = 1e6
    for i in range(len(contour)):
        max_x = max(contour[i][0], max_x)
        min_x = min(contour[i][0], min_x)
        max_y = max(contour[i][1], max_y)
        min_y = min(contour[i][1], min_y)
    top_left = (min_x, max_y)
    top_right = (max_x, max_y)
    bottom_left = (min_x, min_y)
    bottom_right = (max_x, min_y)
    corners.append(top_left)
    corners.append(top_right)
    corners.append(bottom_left)
    corners.append(bottom_right)
    return corners

def get_edge_points(image):
    image = image.copy()
    height, width = image.shape
    pad = 10
    image = image[pad:height - pad, pad:width - pad]
    height, width = image.shape
    contours = []
    visited = {}
    def dfs(image, sp_x, sp_y, to_replace, replace_with):
        height, width = image.shape
        parent_map = {}
        length = 0
        last = None
        stack = [(sp_x, sp_y, 0)]
        parent_map[(sp_x, sp_y)] = None
        while stack:
            x, y, it = stack.pop()
            if image[x, y] != to_replace:
                continue
            image[x, y] = replace_with
            it += 1
            if it > length:
                length = it
                last = (x, y)
            indices = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
            for dx, dy in indices:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and image[nx, ny] == to_replace:
                    if (nx, ny) not in parent_map:
                        parent_map[(nx, ny)] = (x, y)
                        stack.append((nx, ny, it))
        points = []
        while last is not None:
            points.append(last)
            last = parent_map[last]
        points.reverse()
        return points
    def bfs(sx, sy):
        nonlocal visited
        to_it = (sx, sy)
        while to_it is not None:
            queue = deque()
            queue.append(to_it)
            to_it = None
            while queue:
                x, y = queue.popleft()
                if visited.get((x, y)):
                    continue
                image[x, y] = 60
                indices = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
                for dx, dy in indices:
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= height or ny < 0 or ny >= width or image[nx, ny] == 60:
                        continue
                    if image[nx, ny] == 255:
                        to_it = (nx, ny)
                        queue.clear()
                        break
                    if not visited.get((nx, ny)):
                        queue.append((nx, ny))
                visited[(x, y)] = True
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
            if not visited.get((x, y)):
                bfs(x, y)
    return contours

def thresholdImage(img, thres, threshold):
    thres = thres.copy()
    row, col = thres.shape
    for i in range(row):
        for j in range(col):
            if img[i, j] < threshold:
                thres[i, j] = 255
    return thres