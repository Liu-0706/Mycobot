# shake_dynamic_xonly_kalman.py
from pymycobot.mycobot import MyCobot
import time
import joblib
import numpy as np
import math
import head
from filterpy.kalman import KalmanFilter

mc = MyCobot('/dev/ttyAMA0', 1000000)
head.initialize()

# load GPR
gpr_models = [joblib.load(f"gpr_models/gpr_joint{i+1}.pkl") for i in range(6)]


start_coords = [200, 100, 140, -180, 0, 0]
end_coords = [200, -100, 140, -180, 0, 0]

#head.mark_start_and_end_point(start_coords,end_coords)

elevated = start_coords[:]
elevated[2] += 40
mc.send_coords(elevated, 40, 1)
time.sleep(3)
mc.send_coords(start_coords, 20, 1)
time.sleep(3)


def degrees_to_radians(degrees):
    return [math.radians(d) for d in degrees]


kf_list = []
for _ in range(6):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.0], [0.0]])
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.P *= 1000.0
    kf.R = 1.0
    kf.Q = np.array([[1.0, 0.0], [0.0, 1.0]]) * 0.01
    kf_list.append(kf)

steps = 20
start = []
for i in range(steps):
    ratio = i / steps
    target = head.linear_interp(start_coords[:3], end_coords[:3], ratio)
    actual = mc.get_coords()
    #start.append(actual[1])
    """
    if start[0] == start[-1]:
        print("start",start)
        continue
    """
    #target = [200, actual[1], 140]
    print(actual,target)
    """
    if not actual or len(actual) < 3:
        print("no data")
        continue
    """
    #e = np.array(actual) - np.array(curent)

    error = [a - t for a, t in zip(actual[:3], target)]
    x_error = error[0]
    features = np.array(target + [x_error]).reshape(1, -1)

    raw_angles = [model.predict(features)[0] for model in gpr_models]
    """
    mc.send_angles(raw_angles, 20)
    time.sleep(0.02)
    # use KalmanFilter
    """
    filtered_angles = []
    for idx, angle in enumerate(raw_angles):
        kf = kf_list[idx]
        kf.predict()
        kf.update(angle)
        filtered_angles.append(kf.x[0, 0])
    mc.send_angles(filtered_angles, 20)
    #time.sleep(0.01)

    """
    angles_rad = degrees_to_radians(filtered_angles)
    mc.send_radians(angles_rad, 20)
    time.sleep(0.02)
    """


final = end_coords[:]
final[2] += 30
final[0] += 30
mc.send_coords(final, 20, 1)
time.sleep(2)

head.close()




"""
#############################
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

gpr_models = [joblib.load(f"gpr_models/gpr_joint{i+1}.pkl") for i in range(6)]

start_coords = [200, 100, 140, 0, 180, 180]
end_coords = [200, -100, 140, 0, 180, 180]

elevated = start_coords[:]
elevated[2] += 40
mc.send_coords(elevated, 40, 1)
time.sleep(2)
mc.send_coords(start_coords, 20, 1)
time.sleep(2)

def degrees_to_radians(degrees):
    return [math.radians(d) for d in degrees]

kf_list = []
for _ in range(6):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.0], [0.0]])
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.P *= 1000.0
    kf.R = 1.0
    kf.Q = np.array([[1.0, 0.0], [0.0, 1.0]]) * 0.01
    kf_list.append(kf)

steps = 50

for i in range(steps):
    ratio = i / steps
    target = head.linear_interp(start_coords[:3], end_coords[:3], ratio)
    actual = mc.get_coords()
    error = [a - t for a, t in zip(actual[:3], target)]

    features = np.array(target + error).reshape(1, -1)
    raw_angles = [model.predict(features)[0] for model in gpr_models]

    filtered_angles = []
    for idx, angle in enumerate(raw_angles):
        kf = kf_list[idx]
        kf.predict()
        kf.update(angle)
        filtered_angles.append(kf.x[0, 0])

    angles_rad = degrees_to_radians(filtered_angles)
    mc.send_radians(angles_rad, 20)
    #time.sleep(0.01)



head.close()
"""


#####################################
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

