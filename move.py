####zhixian#####
from pymycobot.mycobot import MyCobot
import time
import math
import csv
import head
mc = MyCobot('/dev/ttyAMA0', 1000000)

head.initialize()


"""
start_coords = [120, 20, 140, 0, 180, 180]
end_coords = [200, 20, 140, 0, 180, 180]
end_coords = random_point(end_coords)
print("end_coords",end_coords)
"""

start_coords = [200, 100, 140, 0, 180, 180]
#start_coords = head.random_point(start_coords)
print("start_coords",start_coords)

end_coords = [200, -100, 140, 0, 180, 180]
#end_coords = head.random_point(end_coords)
print("end_coords",end_coords)

join_data = []

#head.record_trajectory(start_coords,end_coords)


elevated = start_coords[:]
elevated[2] += 40
mc.send_coords(elevated, 40, 1)
time.sleep(3)
mc.send_coords(start_coords, 20, 1)
time.sleep(2)

mc.send_coords(end_coords, 15, 1)
time.sleep(4)

final = end_coords[:]
final[2] += 30
final[0] += 30
mc.send_coords(final, 20, 1)
time.sleep(2)

#actual_end_coords = mc.get_coords()
#angles = mc.get_angles()

#save(end_coords, actual_end_coords, angles)

print("end move.py")
head.close()




