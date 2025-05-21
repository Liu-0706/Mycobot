from pymycobot.mycobot import MyCobot
import time
import math
import random
import head
from head import close
from head import save
from head import initialize
from head import add_jitter
from head import random_end

mc = MyCobot('/dev/ttyAMA0', 1000000)

initialize()

start_coords = [120, 20, 140, 0, 180, 180]
end_coords = [200, 20, 140, 0, 180, 180]

head.record_trajectory(start_coords,end_coords)

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
