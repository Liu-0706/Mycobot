import cv2
import numpy as np
def calculate_line_straightness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("no image")   
    # image preprocessing
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)  
    # edges detection
    edges = cv2.Canny(binary, 50, 150)
    """
    cv2.imshow("Edges Detection", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    # straight line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    if lines is None:
        return 0
    
    # angular deviation
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
    
    # Standard Deviation
    angle_std = np.std(angles)
    score = max(0, 100 - angle_std * 5)
    return round(score, 2)

image_path = "curve.jpg"
score = calculate_line_straightness(image_path)
print(f"score of curve: {score}%")

image_path = "straight.jpg"
score = calculate_line_straightness(image_path)
print(f"score of straight: {score}%")

image_path = "straight0.jpg"
score = calculate_line_straightness(image_path)
print(f"score of straight0: {score}%")

image_path = "straight1.jpg"
score = calculate_line_straightness(image_path)
print(f"score of straight1: {score}%")

image_path = "straight2.jpg"
score = calculate_line_straightness(image_path)
print(f"score of straight2: {score}%")

image_path = "straight3.jpg"
score = calculate_line_straightness(image_path)
print(f"score of straight3: {score}%")