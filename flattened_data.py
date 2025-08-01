import pandas as pd
import ast

# 输入输出路径
input_file = "data_1.csv"          # 替换为你的路径
output_file = "flattened_1.csv"   # 输出文件

# 读取原始 CSV
df = pd.read_csv(input_file)

# 新列名（每个 list 的 12 个值对应这些列）
columns = ["x_target", "y_target", "z_target",
           "x_error", "y_error", "z_error",
           "j1", "j2", "j3", "j4", "j5", "j6"]

# 解析函数：将字符串形式的 list 转为真正的 list
def parse_list(cell):
    try:
        return ast.literal_eval(cell) if isinstance(cell, str) else cell
    except:
        return None

# 存放展开后的数据
flattened_rows = []

# 遍历每一行和每一列，提取其中的 list
for i, row in df.iterrows():
    for cell in row:
        values = parse_list(cell)
        if values and len(values) == 12:  # 只处理长度为12的list
            flattened_rows.append(values)
        else:
            print("####################",len(values))
            print("i",i)
            print("cell",cell)

    print("i",i)

# 生成新的 DataFrame
flattened_df = pd.DataFrame(flattened_rows, columns=columns)

# 转换为 float 类型（方便后续分析）
flattened_df = flattened_df.apply(pd.to_numeric, errors='coerce')

# 保存到新 CSV
flattened_df.to_csv(output_file, index=False)

print(f"save as {output_file}")
