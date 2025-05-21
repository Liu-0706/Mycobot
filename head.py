from pymycobot.mycobot import MyCobot
import time
import math
import csv
import random

mc = MyCobot('/dev/ttyAMA0', 1000000)

def close():
    mc.send_coords([60.5, 21.0, 161.0, -143.89, -30.94, 38.33], 30, 1)
    time.sleep(3)
    print("power off")
    mc.power_off()

def initialize():
    mc.power_on()
    if mc.is_controller_connected():
        print("myCobot 280 已连接")
    else:
        print("无法连接到myCobot 280，请检查端口或连接方式")
        return False
    return True

def save(end_coords, actual_end_coords, angles):
    data = end_coords[:3] + actual_end_coords[:3] + angles
    print("data",data)
    save_csv(data)

def save_csv(data):
    with open("data.csv","a",newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(data)

def add_jitter(coords, max_jitter=10):
    return [coord + random.uniform(-max_jitter,max_jitter) for coord in coords]

def random_end(end = [200,20,140,0,180,180], max_jitter=10):
    return [200 + random.uniform(-max_jitter,max_jitter),20,140,0,180,180]
    #end[0] +=  random.uniform(-max_jitter,max_jitter)
    #eturn end

#对两个向量（如坐标）进行线性插值
def linear_interp(start, end, ratio):
    return [s + (e - s) * ratio for s, e in zip(start, end)]

def record_trajectory(start, end, steps=20, interval=0.1):
    # Move to start
    mc.send_coords(start, 50, 1)
    time.sleep(2)
    mc.send_coords(end, 20, 1)
    data = []

    for i in range(steps):
        ratio = i / steps
        expected = linear_interp(start[:3], end[:3], ratio)

        actual = mc.get_coords()
        joints = mc.get_angles()

        error = [a - e for a, e in zip(actual[:3], expected)]

        row = expected + error + joints
        data.append(row)

        time.sleep(interval)
    save_csv(data)