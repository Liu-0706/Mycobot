import cv2
import numpy as np
import os

def detect_coin_diameter(img, known_diameter_mm=25.75):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Gray Image", gray)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=50,
        minRadius=50,
        maxRadius=100
    )

    largest_circle = None
    mm_per_pixel = None

    if circles is not None and len(circles) > 0:
        circles = np.round(circles[0, :]).astype("int")
        largest_circle = max(circles, key=lambda c: c[2])  # c[2] is radius
        x, y, r = largest_circle
        pixel_diameter = 2 * r
        mm_per_pixel = known_diameter_mm / pixel_diameter
    else:
        print("No calibration circle detected in current image.")

    return largest_circle, mm_per_pixel


def visualize_and_compute_deviation(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read {image_path}")
        return None

    largest_circle, mm_per_pixel = detect_coin_diameter(img)
    if largest_circle is None or mm_per_pixel is None:
        print(f"Skipping {image_path}: No circle detected.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print(f"No contour found in {image_path}")
        return None

    contour = max(contours, key=cv2.contourArea)
    points = contour.squeeze()
    if len(points.shape) != 2 or points.shape[0] < 2:
        print(f"Contour in {image_path} is invalid.")
        return None

    # Find A and B
    max_dist = 0
    A, B = points[0], points[0]
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > max_dist:
                A, B = points[i], points[j]
                max_dist = dist

    AB = B - A
    AB_norm = np.linalg.norm(AB)
    if AB_norm == 0:
        return None

    distances = [np.linalg.norm(np.cross(AB, P - A)) / AB_norm for P in points]
    avg_deviation = np.mean(distances) * mm_per_pixel
    print("mm_per_pixel", mm_per_pixel)

    img_vis = img.copy()
    cv2.line(img_vis, tuple(A), tuple(B), (0, 0, 255), 8)
    for P in points:
        cv2.circle(img_vis, tuple(P), 1, (0, 255, 0), -1)
    cv2.circle(img_vis, tuple(A), 6, (255, 0, 255), -1)
    cv2.circle(img_vis, tuple(B), 6, (255, 255, 0), -1)
    cv2.putText(img_vis, 'A', tuple(A + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(img_vis, 'B', tuple(B + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(img_vis, f"Avg Dev: {avg_deviation:.2f}mm", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)

    x, y, r = largest_circle
    cv2.circle(img_vis, (x, y), r, (0, 255, 0), 3)
    cv2.circle(img_vis, (x, y), 2, (0, 0, 255), 3)

    cv2.imwrite(save_path, img_vis)
    print(f"Saved AB deviation image to {save_path}")

    return avg_deviation


list_error_GPR_KF = []
list_error_no_model = []
list_error_only_gpr = []

for i in range(20):
    img_name = f"{i+1}.jpg"

    path_gpr = os.path.join("draw/GPR and KF", img_name)
    path_no_model = os.path.join("draw/no model", img_name)
    path_only_gpr = os.path.join("draw/only GPR", img_name)

    err_gpr = visualize_and_compute_deviation(path_gpr, f"vis_GPR_KF{i+1}.jpg")
    err_no_model = visualize_and_compute_deviation(path_no_model, f"vis_no_model{i+1}.jpg")
    err_only_gpr = visualize_and_compute_deviation(path_only_gpr, f"vis_only_gpr{i+1}.jpg")


    if err_gpr is not None:
        list_error_GPR_KF.append(err_gpr)
    if err_no_model is not None:
        list_error_no_model.append(err_no_model)

    if err_only_gpr is not None:
        list_error_only_gpr.append(err_only_gpr)

with open('error_GPR_KF.txt', 'w') as f:
    f.write(' '.join(map(str, list_error_GPR_KF)))

with open('error_no_model.txt', 'w') as f:
    f.write(' '.join(map(str, list_error_no_model)))

with open('error_only_gpr.txt', 'w') as f:
    f.write(' '.join(map(str, list_error_only_gpr)))


"""
import cv2
import numpy as np

def detect_coin_diameter(img, known_diameter_mm=25.75):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=40,
        minRadius=30,
        maxRadius=250
    )

    pixel_diameter = None
    mm_per_pixel = None

    #Select the largest circle
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        largest_circle = max(circles, key=lambda c: c[2])  # c[2] is the radius r
        x, y, r = largest_circle
        pixel_diameter = 2 * r
        mm_per_pixel = known_diameter_mm / pixel_diameter

    return largest_circle, mm_per_pixel

def visualize_and_compute_deviation(image_path, save_path):
    img = cv2.imread(image_path)
    largest_circle, mm_per_pixel = detect_coin_diameter(img)
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
    avg_deviation = np.mean(distances) * mm_per_pixel
    print("mm_per_pixel",mm_per_pixel)
    # draw
    img_vis = cv2.imread(image_path)
    cv2.line(img_vis, tuple(A), tuple(B), (0, 0, 255), 8)
    for P in points:
        cv2.circle(img_vis, tuple(P), 1, (0, 255, 0), -1)
    cv2.circle(img_vis, tuple(A), 6, (255, 0, 255), -1)
    cv2.circle(img_vis, tuple(B), 6, (255, 255, 0), -1)
    cv2.putText(img_vis, 'A', tuple(A + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(img_vis, 'B', tuple(B + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(img_vis, f"Avg Dev: {avg_deviation:.2f}mm", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 6)
    x, y, r = largest_circle
    cv2.circle(img_vis, (x, y), r, (0, 255, 0), 3)  # Outer ring green
    cv2.circle(img_vis, (x, y), 2, (0, 0, 255), 3)  # Center red
    cv2.imwrite(save_path, img_vis)
    print(f"Saved AB deviation image to {save_path}")
    
    return avg_deviation

#visualize_and_compute_deviation("1.jpg", "1_error.jpg")

list_error_GPR_KF = []
list_error_no_model = []
for i in range(20):
    img = str(i+1) + ".jpg"
    error_GPR_KF = visualize_and_compute_deviation("draw/GPR and KF/" + img, "vis_GPR_KF" + str(i+1) + ".jpg")
    error_no_model = visualize_and_compute_deviation("draw/no model/"  + img, "vis_no_model" + str(i+1) + ".jpg")
    print("deviation_move:", error_GPR_KF)
    print("deviation_move_gpr:", list_error_no_model)

with open('error_GPR_KF.txt', 'w') as f:
    f.write(' '.join(map(str, list_error_GPR_KF)))

with open('error_no_model.txt', 'w') as f:
    f.write(' '.join(map(str, list_error_no_model)))

"""