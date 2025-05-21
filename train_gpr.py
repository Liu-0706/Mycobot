import pandas as pd
import numpy as np
import joblib
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
<<<<<<< HEAD
import sklearn

print(sklearn.__version__)
=======
>>>>>>> Add files manually copied into repo

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
