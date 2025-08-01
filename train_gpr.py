# gpr_train.py
import os
import csv
import ast
import pandas as pd
import numpy as np
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

RAW_CSV = "data_1.csv"
FLAT_CSV = "flattened_1.csv"
MODEL_DIR = "gpr_models_1"
os.makedirs(MODEL_DIR, exist_ok=True)

def flatten_data(raw_csv=RAW_CSV, flat_csv=FLAT_CSV):

    flat_rows = []
    with open(raw_csv, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            for item in row:
                point = ast.literal_eval(item)
                flat_rows.append(point)

    columns = ["x_target", "y_target", "z_target", 
               "x_error", "y_error", "z_error", 
               "j1", "j2", "j3", "j4", "j5", "j6"]

    df = pd.DataFrame(flat_rows, columns=columns)
    df.dropna(inplace=True)  # 清除NaN数据
    df.to_csv(flat_csv, index=False)
    return df

def train_gpr_models(df, model_dir=MODEL_DIR):
    df.dropna(inplace=True)
    #X = df[["x_target", "y_target", "z_target", "x_error", "y_error", "z_error"]].values
    X = df[["x_target", "y_target", "z_target", "x_error"]].values
    Y = df[["j1", "j2", "j3", "j4", "j5", "j6"]].values
    for i in range(6):
        kernel = RBF(length_scale=5.0) + WhiteKernel(noise_level=0.1)
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gpr.fit(X, Y[:, i])
        model_path = os.path.join(model_dir, f"gpr_joint{i+1}.pkl")
        joblib.dump(gpr, model_path)

df = pd.read_csv(FLAT_CSV)
train_gpr_models(df)
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import sklearn

print(sklearn.__version__)

CSV_FILE = "data.csv"
MODEL_DIR = "gpr_models"
os.makedirs(MODEL_DIR, exist_ok=True)


df = pd.read_csv(CSV_FILE)


X = df[["x_target", "y_target", "z_target"]].values
Y = df[["j1", "j2", "j3", "j4", "j5", "j6"]].values



models = []
for i in range(6):
    kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr.fit(X, Y[:, i])
    models.append(gpr)
    joblib.dump(gpr, os.path.join(MODEL_DIR, f"gpr_joint{i+1}.pkl"))

"""