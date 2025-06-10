import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_and_compute_deviation(image_path, save_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print(f"No contour found in {image_path}")
        return float('inf')

    contour = max(contours, key=cv2.contourArea)
    points = contour.squeeze()
    if len(points.shape) != 2 or points.shape[0] < 2:
        print(f"Contour in {image_path} is invalid.")
        return float('inf')

    # Find the two points with the longest distance as A-B
    max_dist = 0
    A, B = points[0], points[0]
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > max_dist:
                A, B = points[i], points[j]
                max_dist = dist

    # calculate deviation
    AB = B - A
    AB_norm = np.linalg.norm(AB)
    if AB_norm == 0:
        return float('inf')

    distances = []
    for P in points:
        AP = P - A
        distance = np.linalg.norm(np.cross(AB, AP)) / AB_norm
        distances.append(distance)
    avg_deviation = np.mean(distances)

    # draw
    img_vis = cv2.imread(image_path)
    cv2.line(img_vis, tuple(A), tuple(B), (0, 0, 255), 8)
    for P in points:
        cv2.circle(img_vis, tuple(P), 1, (0, 255, 0), -1)
    cv2.circle(img_vis, tuple(A), 6, (255, 0, 255), -1)
    cv2.circle(img_vis, tuple(B), 6, (255, 255, 0), -1)
    cv2.putText(img_vis, 'A', tuple(A + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(img_vis, 'B', tuple(B + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imwrite(save_path, img_vis)
    print(f"Saved AB deviation image to {save_path}")

    return avg_deviation

error_move = visualize_and_compute_deviation("move.jpg", "vis_move_labeled.jpg")
error_shake = visualize_and_compute_deviation("shake.jpg", "vis_shake_labeled.jpg")
print("deviation_move:", error_move)
print("deviation_shake:", error_shake)

"""
def visualize_with_furthest_AB(image_path, save_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print(f"No contour found in {image_path}")
        return

    contour = max(contours, key=cv2.contourArea)
    points = contour.squeeze()
    if len(points.shape) != 2 or points.shape[0] < 2:
        print(f"Contour in {image_path} is invalid.")
        return

    # Find furthest pair of points
    max_dist = 0
    A, B = points[0], points[0]
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > max_dist:
                A, B = points[i], points[j]
                max_dist = dist

    # Draw on image
    img_vis = cv2.imread(image_path)
    cv2.line(img_vis, tuple(A), tuple(B), (0, 0, 255), 8)

    for P in points:
        cv2.circle(img_vis, tuple(P), 1, (0, 255, 0), -1)

    # Mark endpoints A and B
    cv2.circle(img_vis, tuple(A), 6, (255, 0, 255), -1)
    cv2.circle(img_vis, tuple(B), 6, (255, 255, 0), -1)
    cv2.putText(img_vis, 'A', tuple(A + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(img_vis, 'B', tuple(B + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imwrite(save_path, img_vis)
    print(f"Saved updated AB visualization to {save_path}")


def calculate_deviation_from_ab_line(contour):
    points = contour.squeeze()
    if len(points.shape) != 2 or points.shape[0] < 2:
        return float('inf')
    
    A = points[0]
    B = points[-1]
    
    AB = B - A
    AB_norm = np.linalg.norm(AB)
    if AB_norm == 0:
        return float('inf')

    distances = []
    for P in points:
        AP = P - A
        distance = np.linalg.norm(np.cross(AB, AP)) / AB_norm
        distances.append(distance)
    
    return np.mean(distances)

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return float('inf')
    
    main_contour = max(contours, key=cv2.contourArea)
    return calculate_deviation_from_ab_line(main_contour)

error_shake = process_image("shake.jpg")
error_move = process_image("move.jpg")

print("error_shake",error_shake)
print("error_move",error_move)

#visualize_with_furthest_AB("move.jpg", "vis_move_labeled.jpg")
#visualize_with_furthest_AB("shake.jpg", "vis_shake_labeled.jpg")
"""