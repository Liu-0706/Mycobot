import cv2
"""
def list_cameras(max_cameras=5):
    print("正在检测摄像头...")
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"yes")
            cap.release()
        else:
            print(f"no")

list_cameras()
"""
cap = cv2.VideoCapture(0)  # 注意这里是1，不是默认0

# 设置分辨率为 1280x720（720p）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()
if ret:
    #cv2.imshow("USB Camera Frame", frame)
    cv2.imwrite("usb_cam_image.jpg", frame)
    print("save as usb_cam_image.jpg")
else:
    print("no image")

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()