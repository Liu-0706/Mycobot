from pymycobot.mycobot import MyCobot
import time
import math
import csv
import random
import joblib
import numpy as np


mc = MyCobot('/dev/ttyAMA0', 1000000)

def close():
    mc.send_coords([60.5, 21.0, 161.0, -143.89, -30.94, 38.33], 30, 1)
    time.sleep(3)
    print("power off")
    mc.power_off()

def initialize():
    mc.power_on()
    if mc.is_controller_connected():
        print("myCobot 280 connected")
    else:
        print("unable to connect myCobot 280")
        return False
    return True

def save(end_coords, actual_end_coords, angles):
    data = end_coords[:3] + actual_end_coords[:3] + angles
    print("data",data)
    with open("data.csv","a",newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(data)

def add_jitter(coords, max_jitter=20):
    return [coord + random.uniform(-max_jitter,max_jitter) for coord in coords]

def random_end(end = [200,20,140,0,180,180], max_jitter=10):
    return [200 + random.uniform(-max_jitter,max_jitter),20,140,0,180,180]
    #end[0] +=  random.uniform(-max_jitter,max_jitter)
    #eturn end

def predict_angles(xyz_target):
    models = [joblib.load(f"gpr_models/gpr_joint{i+1}.pkl") for i in range(6)]
    input_feat = np.array(xyz_target).reshape(1, -1)
    predicted = [model.predict(input_feat)[0] for model in models]
    return predicted