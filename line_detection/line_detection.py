import cv2
import numpy as np

def visualize_and_compute_deviation(image_path, save_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print(f"No contour found in {image_path}")
        return 0.0

    contour = max(contours, key=cv2.contourArea)
    points = contour.squeeze()
    if len(points.shape) != 2 or points.shape[0] < 2:
        print(f"Contour in {image_path} is invalid.")
        return 0.0

    # Find the two points with the longest distance as A-B
    max_dist = 0
    A, B = points[0], points[0]
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > max_dist:
                A, B = points[i], points[j]
                max_dist = dist
    #Calculate the deviation of each point from AB
    AB = B - A
    AB_norm = np.linalg.norm(AB) #The length of the AB vector
    if AB_norm == 0:
        return 0.0

    distances = []
    for P in points:
        AP = P - A
        distance = np.linalg.norm(np.cross(AB, AP)) / AB_norm # Vector Cross Product
        distances.append(distance)
    avg_deviation = np.mean(distances)

    # Percent score
    #Assuming that the semicircle is the least straight, take the radius
    deviation_percent = (avg_deviation / (AB_norm/2)) * 100
    straightness_score = max(0.0, round(100 - deviation_percent, 2))  

    # Draw visualization
    img_vis = cv2.imread(image_path)
    cv2.line(img_vis, tuple(A), tuple(B), (0, 0, 255), 8)
    for P in points:
        cv2.circle(img_vis, tuple(P), 1, (0, 255, 0), -1)
    cv2.circle(img_vis, tuple(A), 6, (255, 0, 255), -1)
    cv2.circle(img_vis, tuple(B), 6, (255, 255, 0), -1)
    cv2.putText(img_vis, 'A', tuple(A + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(img_vis, 'B', tuple(B + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(img_vis, f"{straightness_score}%", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imwrite(save_path, img_vis)
    print(f"Saved visual to {save_path}")
    return straightness_score


score_move = visualize_and_compute_deviation("move.jpg", "vis_move_scored.jpg")
score_shake = visualize_and_compute_deviation("move_gpr.jpg", "vis_move_gpr_scored.jpg")
print(f"move.jpg: {score_move}%")
print(f"shake.jpg: {score_shake}%")

