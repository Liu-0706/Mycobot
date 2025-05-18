import joblib
import numpy as np

# 加载所有模型
models = [joblib.load(f"gpr_models/gpr_joint{i+1}.pkl") for i in range(6)]

def predict_angles(xyz_target):
    """输入目标点 [x, y, z]，返回预测的6个关节角度列表"""
    input_feat = np.array(xyz_target).reshape(1, -1)
    predicted = [model.predict(input_feat)[0] for model in models]
    return predicted

# 示例目标点
target_xyz = [200, 20, 140]
angles = predict_angles(target_xyz)

print("angles:", angles)