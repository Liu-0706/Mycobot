# shake_dynamic.py
from pymycobot.mycobot import MyCobot
import time
import joblib
import numpy as np
import head

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

# 动态绘制并补偿
steps = 100
prev_angles = mc.get_angles()
alpha = 0.7  # 平滑系数

for i in range(steps):
    ratio = i / steps
    target = head.linear_interp(start_coords[:3], end_coords[:3], ratio)
    actual = mc.get_coords()
    error = [a - t for a, t in zip(actual[:3], target)]

    features = np.array(target + error).reshape(1, -1)
    raw_angles = [model.predict(features)[0] for model in gpr_models]

    # 平滑处理角度输出
    smoothed_angles = [
        alpha * new + (1 - alpha) * old
        for new, old in zip(raw_angles, prev_angles)
    ]

    mc.send_angles(smoothed_angles, 40)
    prev_angles = smoothed_angles

    # 动态等待直到基本到位
    while not head.is_in_position(smoothed_angles):
        time.sleep(0.01)

# 提笔动作
final = end_coords[:]
final[2] += 30
mc.send_coords(final, 20, 1)
while not mc.is_in_position(final, 2):
    time.sleep(0.01)

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

