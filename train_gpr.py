# gpr_train.py
import os
import csv
import ast
import pandas as pd
import numpy as np
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

RAW_CSV = "data_new.csv"
FLAT_CSV = "flattened_data.csv"
MODEL_DIR = "gpr_models"
os.makedirs(MODEL_DIR, exist_ok=True)

""""""
# ========== Step 1: 扁平化处理 ==========
print("开始扁平化数据处理...")

flat_rows = []
with open(RAW_CSV, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        for item in row:
            point = ast.literal_eval(item)  # 转换为列表对象
            flat_rows.append(point)

columns = ["x_target", "y_target", "z_target", 
           "x_error", "y_error", "z_error", 
           "j1", "j2", "j3", "j4", "j5", "j6"]

df = pd.DataFrame(flat_rows, columns=columns)
df.to_csv(FLAT_CSV, index=False)


# ========== Step 2: 训练 GPR 模型 ==========
print("开始训练 GPR 模型...")
df.dropna(inplace=True)
X = df[["x_target", "y_target", "z_target", "x_error", "y_error", "z_error"]].values
Y = df[["j1", "j2", "j3", "j4", "j5", "j6"]].values

for i in range(6):
    kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr.fit(X, Y[:, i])
    model_path = os.path.join(MODEL_DIR, f"gpr_joint{i+1}.pkl")
    joblib.dump(gpr, model_path)
    print(f"已训练并保存模型: {model_path}")

print("所有 GPR 模型训练完成！")

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

# 正确读取包含表头的数据
df = pd.read_csv(CSV_FILE)

# 输入/输出数据
X = df[["x_target", "y_target", "z_target"]].values
Y = df[["j1", "j2", "j3", "j4", "j5", "j6"]].values


# ===== 训练模型 =====
print("开始训练 GPR 模型...")
models = []
for i in range(6):
    kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr.fit(X, Y[:, i])
    models.append(gpr)
    joblib.dump(gpr, os.path.join(MODEL_DIR, f"gpr_joint{i+1}.pkl"))
    print(f"已保存：gpr_joint{i+1}.pkl")

print("所有模型训练完成！")
"""