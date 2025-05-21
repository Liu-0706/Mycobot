from pymycobot.mycobot import MyCobot
import time
import math
import random
from head import close
from head import save
from head import initialize
from head import add_jitter
from head import random_end
from head import predict_angles

mc = MyCobot('/dev/ttyAMA0', 1000000)

initialize()

start_coords = [120, 20, 140, 0, 180, 180]
#end_coords = [200, 20, 140, 0, 180, 180]

end_coords = random_end()
print("end_coords",end_coords)
end_coords = add_jitter(end_coords)
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
    end_angles = predict_angles(end_coords[:3])
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
close()


