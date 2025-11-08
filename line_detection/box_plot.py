import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_txt(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    txt = p.read_text().strip()
    if not txt:
        return np.array([])
    return np.fromstring(txt.replace(",", " "), sep=" ")

data_gpr_kf   = load_txt("error_GPR_KF.txt")
data_only_gpr = load_txt("error_only_gpr.txt")
data_nomodel  = load_txt("error_no_model.txt")

data_list = [data_gpr_kf, data_only_gpr, data_nomodel]
labels    = ["With GPR+KF", "Only GPR", "Without Model"]

plt.figure(figsize=(7.2, 5.2))
box = plt.boxplot(
    data_list,
    labels=labels,
    patch_artist=True,
    boxprops=dict(color='blue', facecolor='white', linewidth=1.5),
    medianprops=dict(color='red', linewidth=1.8),
    whiskerprops=dict(color='gray', linestyle='--', linewidth=1.2),
    capprops=dict(color='gray', linewidth=1.2),
    flierprops=dict(marker='o', markersize=4, markerfacecolor='white', markeredgecolor='gray')
)

xpos = np.arange(1, len(data_list) + 1)
means = [np.mean(d) if len(d) else np.nan for d in data_list]
sizes = [len(d) for d in data_list]

plt.scatter(xpos, means, marker='D', s=60, edgecolor='black', facecolor='yellow', zorder=5, label='Mean')



plt.title("Comparison of X-axis Deviation")
plt.ylabel("Average deviation (mm)")
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.legend(loc='upper left')
plt.tight_layout()

plt.show()

for name, arr in zip(labels, data_list):
    if len(arr):
        print(f"{name:>15s} | n={len(arr):2d} | mean={np.mean(arr):.2f} | median={np.median(arr):.2f} | "
              f"Q1={np.percentile(arr,25):.2f} | Q3={np.percentile(arr,75):.2f} | min={np.min(arr):.2f} | max={np.max(arr):.2f}")
    else:
        print(f"{name:>15s} | n=0 (empty)")
