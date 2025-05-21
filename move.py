####zhixian#####
from pymycobot.mycobot import MyCobot
import time
import math
import csv
from head import close
from head import save
from head import initialize
from head import random_end
# 初始化MyCobot（根据你的实际串口修改）
mc = MyCobot('/dev/ttyAMA0', 1000000)

initialize()

# 定义初始点和目标点
start_coords = [120, 20, 140, 0, 180, 180]
end_coords = [200, 20, 140, 0, 180, 180]
end_coords = random_point(end_coords)
print("end_coords",end_coords)

join_data = []
# 1. 移动到初始点
mc.send_coords(start_coords, 50, 1)
time.sleep(2)

"""
# 2. 移动到目标点
for _ in range(20):
    mc.send_coords(end_coords, 20, 1)
    time.sleep(0.01)
    join_data.append(mc.get_angles())

print("join_data",join_data)
"""
mc.send_coords(end_coords, 20, 1)
time.sleep(2)

actual_end_coords = mc.get_coords()
angles = mc.get_angles()

save(end_coords, actual_end_coords, angles)

print("end move.py")
close()




