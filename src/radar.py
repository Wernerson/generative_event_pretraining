# radar_10metrics_units_grouped.py
import numpy as np
import matplotlib.pyplot as plt

def wrap_label(s):
    # 把单位单独放到下一行
    s = s.replace(' (', '\n(').replace(' [', '\n[')
    # 其余空格再换行
    return s.replace(' ', '\n')

# 原始顺序：seg4 + recog4 + depth2
labels_base = [
    'DDD17 mIoU [%]', 'DDD17 mAcc [%]', 'DSEC mIoU [%]', 'DSEC mAcc [%]',
    'N-ImageNet acc1 [%]', 'N-ImageNet acc5 [%]',
    'N-Caltech101 acc1 [%]', 'N-Caltech101 acc5 [%]',
    'MVSEC Abs (m)↓', 'MVSEC RMS (m)↓'
]
# 高越好 or 低越好
higher_is_better_base = [True, True, True, True, True, True, True, True, False, False]

# 原始数值（按你的表）
raw = {
    'ECDP [37]':   [54.66, 66.08, 52.52, 60.55, 64.83,  86.30,  87.66,      None,  4.49, 7.68],
    'ECDDP [38]':  [55.73, 64.77, 61.25, 69.62, None,   None,   None,       None,  3.99, 6.96],
    'DINOv2 [23]': [53.85, 64.50, 52.17, 59.80, 60.80,  83.97,  91.94,      98.12,  4.45, 7.65],
    'Ours':        [60.01, 71.45, 65.22, 74.66, 65.11,  87.36,  93.05,      98.57,  3.85, 6.60],
}
methods = list(raw.keys())

order = [0, 1, 2, 3, 8, 9, 7, 5, 6, 4]
labels = [labels_base[i] for i in order]
higher_is_better = [higher_is_better_base[i] for i in order]
for m in methods:
    raw[m] = [raw[m][i] for i in order]

labels_wrapped = [wrap_label(s) for s in labels]

num_metrics = len(labels)
norm = {m: [] for m in methods}

for j in range(num_metrics):
    higher = higher_is_better[j]
    def score(v):  # 将“低越好”转换为“高越好”
        return v if higher else -v

    vals_scores = [score(raw[m][j]) for m in methods if raw[m][j] is not None]
    if not vals_scores:
        for m in methods:
            norm[m].append(0.0)
        continue

    min_val = min(vals_scores)  # 最差
    max_val = max(vals_scores)  # 最好
    adj_min = min_val - 0.1 * abs(min_val)

    ours_raw = raw['Ours'][j]
    ours_val = score(ours_raw) if ours_raw is not None else None
    denom = max((ours_val - adj_min) if ours_val is not None else 0.0, 1e-8)

    for m in methods:
        v = raw[m][j]
        if v is None:
            norm[m].append(0.0)
        else:
            n = (score(v) - adj_min) / denom
            norm[m].append(float(np.clip(n, 0.0, 1.0)))

angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(7.6, 7.6))
ax = plt.subplot(111, polar=True)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels_wrapped, fontsize=11)
ax.set_rlabel_position(90)
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_yticklabels(['0', '0.5', '1.0'], fontsize=10)
ax.grid(alpha=1, linewidth=1)
ax.spines['polar'].set_linewidth(1.0)
plt.ylim(0, 1.05)

color_map = {
    'Ours':        'tab:red',
    'DINOv2 [23]': 'C2',
    'ECDDP [38]':  'C0',
    'ECDP [37]':   'C1',
}

fill_alpha_main = 0.2
fill_alpha_others = 0.2

for m in methods:
    data = norm[m] + norm[m][:1]
    c = color_map[m]
    if m == 'Ours':
        ax.plot(angles, data, linewidth=3, marker='o', markersize=5, color=c, label=m, zorder=3)
        ax.fill(angles, data, alpha=fill_alpha_main, color=c, zorder=2)
    else:
        ax.plot(angles, data, linewidth=1.8, marker='o', markersize=5, linestyle='--', color=c, label=m, zorder=1)
        ax.fill(angles, data, alpha=fill_alpha_others, color=c, zorder=0)

plt.legend(
    loc='lower center', bbox_to_anchor=(0.5, -0.18),
    ncol=4, fontsize=11, frameon=False, handlelength=2.6
)

plt.tight_layout()
plt.savefig('radar.png', dpi=400, bbox_inches='tight')
plt.savefig('radar.pdf', dpi=400, bbox_inches='tight')
plt.show()
