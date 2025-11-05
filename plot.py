import pandas as pd
import matplotlib.pyplot as plt


file_path = 'flattened_data.csv'
df = pd.read_csv(file_path)


group_size = 17
num_groups = len(df) // group_size
print("len(df)",len(df))
print("num_groups ",num_groups+1)


plt.figure(figsize=(12, 6))

for i in range(num_groups):
    start = i * group_size
    end = start + group_size
    group = df.iloc[start:end].copy()
    y0 = group['y_target'].iloc[0]
    relative_y = (group['y_target'] - y0).abs()
    plt.plot(relative_y, group['x_error'], label=f'Group {i+1}')

plt.xlabel('Absolute Distance from Group Start (|y_target - y0|)')
plt.ylabel('x_error')
plt.title('x_error vs. Absolute y_target Distance (17-point segments)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

