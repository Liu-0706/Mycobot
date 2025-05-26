from pymycobot.mycobot import MyCobot
import time
import math
import random
import head

mc = MyCobot('/dev/ttyAMA0', 1000000)

head.initialize()


start_coords = [200, 100, 140, 0, 180, 180]
start_coords = head.random_point(start_coords)
print("start_coords",start_coords)

end_coords = [200, -100, 140, 0, 180, 180]
end_coords = head.random_point(end_coords)
print("end_coords",end_coords)



start_coords[2] += 40
mc.send_coords(start_coords, 40, 1)
time.sleep(3)
start_coords[2] -= 40
mc.send_coords(start_coords, 20, 1)
time.sleep(3)

mc.send_coords(end_coords, 20, 1)
time.sleep(3)

print("end data_collection.py")
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

