import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体（如果需要）
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 数据定义
data = {
    'Model': [
        'energy2d_mse_2ch',
        'energy2d_multi_2ch', 
        'freq2d_(a)_mse_2ch',
        'freq2d_(a)_multi_3ch',
        'freq2d_(b)_mse',
        'freq2d_(b)_multi',
        'freq2d_(c)_mse',
        'freq2d_(c)_multi',
        'simp2d_mse_4ch',
        'simp2d_multi_4ch'
    ],
    'ΔPSNR': [1.46, 1.59, 1.77, 2.57, 2.34, 2.60, 2.19, 2.72, 2.61, 2.62],
    'ΔPSNR_std': [0.51, 0.52, 0.26, 0.13, 0.26, 0.11, 0.35, 0.12, 0.11, 0.15],
    'Mag_MSE': [0.188, 0.185, 0.181, 0.140, 0.171, 0.145, 0.174, 0.134, 0.153, 0.138],
    'Mag_MSE_std': [0.000743, 0.00422, 0.00324, 0.00547, 0.00969, 0.00458, 0.0118, 0.00345, 0.00827, 0.00399],
    'Phase_MSE': [0.237, 0.234, 0.229, 0.175, 0.216, 0.181, 0.220, 0.167, 0.194, 0.171],
    'Phase_MSE_std': [0.000928, 0.00503, 0.00394, 0.00788, 0.0123, 0.00657, 0.0148, 0.00496, 0.0115, 0.00568],
    'Loss_Type': ['MSE', 'Multi', 'MSE', 'Multi', 'MSE', 'Multi', 'MSE', 'Multi', 'MSE', 'Multi']
}

df = pd.DataFrame(data)

# 颜色方案
colors_mse = '#3498db'      # 蓝色 for MSE loss
colors_multi = '#e74c3c'    # 红色 for Multi loss

def get_colors(loss_types):
    return [colors_mse if lt == 'MSE' else colors_multi for lt in loss_types]

# 创建图形
fig = plt.figure(figsize=(16, 12))

# ==================== Plot 1: ΔPSNR Bar Chart with Error Bars ====================
ax1 = fig.add_subplot(2, 2, 1)
x = np.arange(len(df['Model']))
colors = get_colors(df['Loss_Type'])

bars = ax1.bar(x, df['ΔPSNR'], yerr=df['ΔPSNR_std'], capsize=4, 
               color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

ax1.set_xlabel('Model', fontsize=11)
ax1.set_ylabel('Average ΔPSNR (dB)', fontsize=11)
ax1.set_title('PSNR Improvement Over Baseline (Higher is Better)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=9)
ax1.axhline(y=df['ΔPSNR'].mean(), color='gray', linestyle='--', alpha=0.7, label=f'Mean: {df["ΔPSNR"].mean():.2f} dB')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, val in zip(bars, df['ΔPSNR']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08, 
             f'{val:.2f}', ha='center', va='bottom', fontsize=8)

# ==================== Plot 2: Mag MSE Bar Chart ====================
ax2 = fig.add_subplot(2, 2, 2)
bars2 = ax2.bar(x, df['Mag_MSE'], yerr=df['Mag_MSE_std'], capsize=4,
                color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

ax2.set_xlabel('Model', fontsize=11)
ax2.set_ylabel('Magnitude MSE', fontsize=11)
ax2.set_title('Frequency Magnitude MSE (Lower is Better)', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=9)
ax2.axhline(y=0.189, color='green', linestyle='--', alpha=0.7, label='Baseline: 0.189')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# ==================== Plot 3: Phase MSE Bar Chart ====================
ax3 = fig.add_subplot(2, 2, 3)
bars3 = ax3.bar(x, df['Phase_MSE'], yerr=df['Phase_MSE_std'], capsize=4,
                color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

ax3.set_xlabel('Model', fontsize=11)
ax3.set_ylabel('Phase MSE', fontsize=11)
ax3.set_title('Frequency Phase MSE (Lower is Better)', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=9)
ax3.axhline(y=0.239, color='green', linestyle='--', alpha=0.7, label='Baseline: 0.239')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# ==================== Plot 4: Scatter Plot - ΔPSNR vs Mag MSE ====================
ax4 = fig.add_subplot(2, 2, 4)

for i, (model, dpsnr, mag_mse, loss_type) in enumerate(zip(df['Model'], df['ΔPSNR'], df['Mag_MSE'], df['Loss_Type'])):
    color = colors_mse if loss_type == 'MSE' else colors_multi
    marker = 'o' if loss_type == 'MSE' else 's'
    ax4.scatter(dpsnr, mag_mse, c=color, marker=marker, s=120, edgecolors='black', linewidths=0.5, alpha=0.8)
    ax4.annotate(model.replace('_', '\n'), (dpsnr, mag_mse), textcoords="offset points", 
                 xytext=(0, 8), ha='center', fontsize=7)

ax4.set_xlabel('ΔPSNR (dB)', fontsize=11)
ax4.set_ylabel('Magnitude MSE', fontsize=11)
ax4.set_title('ΔPSNR vs Magnitude MSE (Bottom-Right is Best)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 添加图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors_mse, label='MSE Loss'),
                   Patch(facecolor=colors_multi, label='Multi Loss')]
ax4.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('model_comparison_overview.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 额外图表: MSE vs MultiLoss 对比 ====================
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

# 将模型按架构分组
architectures = ['energy2d', 'freq2d_(a)', 'freq2d_(b)', 'freq2d_(c)', 'simp2d']
mse_dpsnr = [1.46, 1.77, 2.34, 2.19, 2.61]
multi_dpsnr = [1.59, 2.57, 2.60, 2.72, 2.62]
mse_mag = [0.188, 0.181, 0.171, 0.174, 0.153]
multi_mag = [0.140, 0.140, 0.145, 0.134, 0.138]
mse_phase = [0.237, 0.229, 0.216, 0.220, 0.194]
multi_phase = [0.175, 0.175, 0.181, 0.167, 0.171]

x_arch = np.arange(len(architectures))
width = 0.35

# ΔPSNR comparison
ax = axes[0]
bars1 = ax.bar(x_arch - width/2, mse_dpsnr, width, label='MSE Loss', color=colors_mse, alpha=0.85)
bars2 = ax.bar(x_arch + width/2, multi_dpsnr, width, label='Multi Loss', color=colors_multi, alpha=0.85)
ax.set_xlabel('Architecture')
ax.set_ylabel('ΔPSNR (dB)')
ax.set_title('ΔPSNR: MSE vs Multi Loss')
ax.set_xticks(x_arch)
ax.set_xticklabels(architectures, rotation=30, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Mag MSE comparison
ax = axes[1]
bars1 = ax.bar(x_arch - width/2, mse_mag, width, label='MSE Loss', color=colors_mse, alpha=0.85)
bars2 = ax.bar(x_arch + width/2, multi_mag, width, label='Multi Loss', color=colors_multi, alpha=0.85)
ax.set_xlabel('Architecture')
ax.set_ylabel('Magnitude MSE')
ax.set_title('Mag MSE: MSE vs Multi Loss (Lower=Better)')
ax.set_xticks(x_arch)
ax.set_xticklabels(architectures, rotation=30, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Phase MSE comparison  
ax = axes[2]
bars1 = ax.bar(x_arch - width/2, mse_phase, width, label='MSE Loss', color=colors_mse, alpha=0.85)
bars2 = ax.bar(x_arch + width/2, multi_phase, width, label='Multi Loss', color=colors_multi, alpha=0.85)
ax.set_xlabel('Architecture')
ax.set_ylabel('Phase MSE')
ax.set_title('Phase MSE: MSE vs Multi Loss (Lower=Better)')
ax.set_xticks(x_arch)
ax.set_xticklabels(architectures, rotation=30, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('mse_vs_multiloss_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 雷达图: 综合性能对比 ====================
fig3 = plt.figure(figsize=(10, 8))

# Top 5 模型的手动数据定义
selected_data = {
    'freq2d_(c)_multi': {'ΔPSNR': 2.72, 'Mag_MSE': 0.134, 'Phase_MSE': 0.167, 'Stability': 0.12},
    'simp2d_multi_4ch': {'ΔPSNR': 2.62, 'Mag_MSE': 0.138, 'Phase_MSE': 0.171, 'Stability': 0.15},
    'freq2d_(b)_multi': {'ΔPSNR': 2.60, 'Mag_MSE': 0.145, 'Phase_MSE': 0.181, 'Stability': 0.11},
    'simp2d_mse_4ch': {'ΔPSNR': 2.61, 'Mag_MSE': 0.153, 'Phase_MSE': 0.194, 'Stability': 0.11},
    'freq2d_(a)_multi_3ch': {'ΔPSNR': 2.57, 'Mag_MSE': 0.140, 'Phase_MSE': 0.175, 'Stability': 0.13}
}

# 归一化数据用于雷达图 (0-1范围,高=好)
def normalize_for_radar(data_dict):
    normalized = {}
    for model, metrics in data_dict.items():
        normalized[model] = {
            'ΔPSNR': (metrics['ΔPSNR'] - 1.0) / (3.0 - 1.0),  # 归一化到0-1
            'Mag_MSE': 1 - (metrics['Mag_MSE'] - 0.13) / (0.19 - 0.13),  # 反转,低=好
            'Phase_MSE': 1 - (metrics['Phase_MSE'] - 0.16) / (0.24 - 0.16),  # 反转,低=好
            'Stability': 1 - (metrics['Stability'] - 0.1) / (0.55 - 0.1)  # 反转,低=好
        }
    return normalized

norm_data = normalize_for_radar(selected_data)

categories = ['ΔPSNR', 'Mag MSE\n(inverted)', 'Phase MSE\n(inverted)', 'Stability\n(inverted)']
num_vars = len(categories)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

ax = fig3.add_subplot(111, polar=True)

colors_radar = plt.cm.Set2(np.linspace(0, 1, len(selected_data)))

for idx, (model, metrics) in enumerate(norm_data.items()):
    values = [metrics['ΔPSNR'], metrics['Mag_MSE'], metrics['Phase_MSE'], metrics['Stability']]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors_radar[idx])
    ax.fill(angles, values, alpha=0.1, color=colors_radar[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('Top 5 Models: Normalized Performance\n(Outer = Better)', fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('radar_chart_top_models.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 打印汇总表格 ====================
print("\n" + "="*80)
print("PERFORMANCE SUMMARY TABLE")
print("="*80)
print(f"{'Model':<25} {'ΔPSNR (dB)':<18} {'Mag MSE':<20} {'Phase MSE':<18}")
print("-"*80)
for _, row in df.sort_values('ΔPSNR', ascending=False).iterrows():
    print(f"{row['Model']:<25} {row['ΔPSNR']:.2f} ± {row['ΔPSNR_std']:.2f}      "
          f"{row['Mag_MSE']:.3f} ± {row['Mag_MSE_std']:.4f}    "
          f"{row['Phase_MSE']:.3f} ± {row['Phase_MSE_std']:.4f}")
print("="*80)
print(f"\n🏆 Best Model: freq2d_(c)_multi")
print(f"   - Highest ΔPSNR: +2.72 dB")
print(f"   - Lowest Mag MSE: 0.134")
print(f"   - Lowest Phase MSE: 0.167")