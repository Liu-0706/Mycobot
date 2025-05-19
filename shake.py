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
end_coords = random_end()
print("end_coords",end_coords)

#end_coords = [200, 20, 140, 0, 180, 180]

def shake(start_coords, end_coords):
    end_coords = add_jitter(end_coords)
    end_coords = predict_angles(end_coords)
    mc.send_coords(start_coords, 50, 1)
    time.sleep(2)
    mc.send_angles(end_coords, 20, 1)
    time.sleep(2)


print("end shake.py")
close()


