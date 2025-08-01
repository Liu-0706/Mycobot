from pymycobot.mycobot import MyCobot
import time
import math
import csv
import random

mc = MyCobot('/dev/ttyAMA0', 1000000)
"""
def close():
    mc.send_coords([60.5, 21.0, 161.0, -143.89, -30.94, 38.33], 30, 1)
    time.sleep(3)
    print("power off")
    mc.power_off()
"""
def close():
    mc.send_angles([92.46, 141.94, -151.25, -24.52, 4.57, 46.66], 30)
    time.sleep(4)
    print("power off")
    mc.power_off()

def move_to_start():
    mc.send_angles([92.46, 141.94, -151.25, -24.52, 4.57, 46.66], 30)
    time.sleep(4)

def initialize():
    mc.power_on()
    if mc.is_controller_connected():
        print("myCobot 280 is connected")
    else:
        print("unable to connect myCobot 280")
        return False
    return True

def save(end_coords, actual_end_coords, angles):
    data = end_coords[:3] + actual_end_coords[:3] + angles
    print("data",data)
    save_csv(data,"data.csv")

def save_csv(data,data_name):
    with open(data_name,"a",newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(data)

def add_jitter(coords, max_jitter=10):
    return [coord + random.uniform(-max_jitter,max_jitter) for coord in coords]

def random_point(point, max_jitter=5):
    point[0] += random.uniform(-max_jitter,max_jitter)
    point[1] += random.uniform(-max_jitter,max_jitter)
    print("point",point)
    return point
    #end[0] +=  random.uniform(-max_jitter,max_jitter)
    #eturn end

#Linear interpolation of two vectors
def linear_interp(start, end, ratio):
    return [s + (e - s) * ratio for s, e in zip(start, end)]

def record_trajectory(start, end, steps=17, interval=0.02):
    # Move to start
    start[2] += 40
    mc.send_coords(start, 40, 1)
    time.sleep(3)
    start[2] -= 40
    mc.send_coords(start, 20, 1)
    time.sleep(3)

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

        #time.sleep(interval)
    end[0] += 20
    end[2] += 40
    mc.send_coords(end, 20, 1)
    time.sleep(2)
    save_csv(data,"data_1.csv")

def is_angles_close(target_angles, threshold=2.0):
    current_angles = mc.get_angles()
    diffs = [abs(c - t) for c, t in zip(current_angles, target_angles)]
    return all(d < threshold for d in diffs)

def mark_start_and_end_point(start_coords, end_coords):
    elevated = start_coords[:]
    elevated[2] += 40
    mc.send_coords(elevated, 40, 1)
    time.sleep(3)
    mc.send_coords(start_coords, 20, 1)
    time.sleep(3)
    move_to_start()

    elevated = end_coords[:]
    elevated[2] += 40
    mc.send_coords(elevated, 40, 1)
    time.sleep(3)
    mc.send_coords(end_coords, 20, 1)
    time.sleep(3)
    move_to_start()


