import cv2
import numpy as np

def crop_black_line(image_path, padding=20):
    # 读取图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 进行二值化处理（黑色线条突出）
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 找到黑色线条的轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("未检测到黑色线条")
        return None
    
    # 找到最大的轮廓（假设黑线是主要的对象）
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 增加额外的白色背景区域
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    # 裁剪图像
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def calculate_line_straightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 预处理图像
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)  
    
    # 边缘检测
    edges = cv2.Canny(binary, 50, 150)
    
    # 显示边缘检测结果
    cv2.imshow("Edges Detection", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    if lines is None:
        return 0
    
    # 计算角度偏差
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
    
    # 计算标准差
    angle_std = np.std(angles)
    score = max(0, 100 - angle_std)
    return round(score, 2)

# 处理图像并计算直线评分
image_path = "shake.jpg"  # 输入图像
cropped_image = crop_black_line(image_path, padding=50)  # 设置额外的白色背景区域大小

if cropped_image is not None:
    score = calculate_line_straightness(cropped_image)
    print(f"直线评分: {score}%")
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("未能检测到黑色线条")

# 处理图像并计算直线评分
image_path = "move.jpg"  # 输入图像
cropped_image = crop_black_line(image_path, padding=50)  # 设置额外的白色背景区域大小

if cropped_image is not None:
    score = calculate_line_straightness(cropped_image)
    print(f"直线评分: {score}%")
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("未能检测到黑色线条")