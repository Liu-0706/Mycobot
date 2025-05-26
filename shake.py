# shake_dynamic_kalman.py
from pymycobot.mycobot import MyCobot
import time
import joblib
import numpy as np
import math
import head
from filterpy.kalman import KalmanFilter

mc = MyCobot('/dev/ttyAMA0', 1000000)
head.initialize()

# 加载 GPR 模型
gpr_models = [joblib.load(f"gpr_models/gpr_joint{i+1}.pkl") for i in range(6)]

# 初始和目标点
start_coords = [200, 100, 140, 0, 180, 180]
end_coords = [200, -100, 140, 0, 180, 180]

# 提升安全高度
elevated = start_coords[:]
elevated[2] += 40
mc.send_coords(elevated, 40, 1)
time.sleep(2)
mc.send_coords(start_coords, 20, 1)
time.sleep(2)

# 辅助函数：角度转弧度
def degrees_to_radians(degrees):
    return [math.radians(d) for d in degrees]

# 初始化卡尔曼滤波器（6个独立滤波器用于6个关节）
kf_list = []
for _ in range(6):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.0], [0.0]])  # 初始状态：位置和速度
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])  # 状态转移矩阵
    kf.H = np.array([[1.0, 0.0]])  # 观测矩阵
    kf.P *= 1000.0  # 初始协方差
    kf.R = 1.0  # 测量噪声
    kf.Q = np.array([[1.0, 0.0], [0.0, 1.0]]) * 0.01  # 过程噪声
    kf_list.append(kf)

# 动态绘制
steps = 100

for i in range(steps):
    ratio = i / steps
    target = head.linear_interp(start_coords[:3], end_coords[:3], ratio)
    actual = mc.get_coords()
    error = [a - t for a, t in zip(actual[:3], target)]

    features = np.array(target + error).reshape(1, -1)
    raw_angles = [model.predict(features)[0] for model in gpr_models]

    # 卡尔曼滤波处理
    filtered_angles = []
    for idx, angle in enumerate(raw_angles):
        kf = kf_list[idx]
        kf.predict()
        kf.update(angle)
        filtered_angles.append(kf.x[0, 0])

    angles_rad = degrees_to_radians(filtered_angles)
    mc.send_radians(angles_rad, 40)
    time.sleep(0.05)

# 提笔动作
final = end_coords[:]
final[2] += 30
mc.send_coords(final, 20, 1)
time.sleep(2)

head.close()




"""
from pymycobot.mycobot import MyCobot
import time
import math
import random
import head

mc = MyCobot('/dev/ttyAMA0', 1000000)

head.initialize()

start_coords = [120, 20, 140, 0, 180, 180]
end_coords = [200, 20, 140, 0, 180, 180]

end_coords = head.random_point(end_coords)  #x和y随机添加参数，生成随机终点
print("end_coords",end_coords)
end_coords = head.add_jitter(end_coords)
print("end_coords_jitter",end_coords)

def shake(start_coords, end_coords):
    start_coords[2] += 30
    mc.send_coords(start_coords, 40, 1)
    time.sleep(2)
    start_coords[2] -= 30
    mc.send_coords(start_coords, 20, 1)
    time.sleep(1)


    mc.send_coords(end_coords, 20, 1)
    time.sleep(2)
    end_coords[2] += 30
    mc.send_coords(end_coords, 20, 1)
    time.sleep(1)


def shake_with_gpr(start_coords, end_coords):
    end_angles = head.predict_angles(end_coords[:3])
    print("end_angles",end_angles)
    start_coords[2] += 30
    mc.send_coords(start_coords, 40, 1)
    time.sleep(2)
    start_coords[2] -= 30
    mc.send_coords(start_coords, 20, 1)
    time.sleep(1)

    mc.send_angles(end_angles, 20)
    time.sleep(2)

shake(start_coords, end_coords)

shake_with_gpr(start_coords, end_coords)

print("end shake.py")
head.close()
"""

