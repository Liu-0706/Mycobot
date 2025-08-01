from pymycobot.mycobot import MyCobot
import time
import math
import random
import head

mc = MyCobot('/dev/ttyAMA0', 1000000)

head.initialize()

for _ in range(20):
    start_coords = [200, 100, 140, 0, 180, 180]
    #start_coords = head.random_point(start_coords)
    print("start_coords",start_coords)

    end_coords = [200, -100, 140, 0, 180, 180]
    #end_coords = head.random_point(end_coords)
    print("end_coords",end_coords)

    head.record_trajectory(start_coords,end_coords)

print("end data_collection.py")
head.close()

"""
def shake(start_coords, end_coords):
    jitter_coodrs = add_jitter(end_coords)

    mc.send_coords(start_coords, 50, 1)
    time.sleep(2)
    mc.send_coords(jitter_coodrs, 20, 1)
    time.sleep(2)

    actual_end_coords = mc.get_coords()
    angles = mc.get_angles()

    save(end_coords, actual_end_coords, angles)

for _ in range(10):
    end_coords = random_end()
    print("end_coords",end_coords)
    for _ in range(20):
        shake(start_coords, end_coords)
print("end shake.py")
close()
"""
