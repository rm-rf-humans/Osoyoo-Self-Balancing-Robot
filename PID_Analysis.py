import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter


df = pd.read_csv('balance_data.csv') 

# 1. Data Overview
print(df.head())  # Display the first few rows of the DataFrame


sns.set(style="whitegrid")

FIGSIZE = (13.33, 7.5)  # inches
DPI = 288               # 13.33*288 ≈ 3840, 7.5*288 ≈ 2160

# 2. Error vs Time
plt.figure(figsize=FIGSIZE, dpi=DPI)
plt.plot(df['t'], df['err'], label='Error', color='r')
plt.axhline(0, color='gray', linestyle='--')
plt.title('PID Error vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Error (°)')
plt.legend()
plt.tight_layout()
plt.savefig('error_vs_time_.png', dpi=DPI, bbox_inches='tight')
plt.close()

# 3. PID Outputs vs Time
plt.figure(figsize=FIGSIZE, dpi=DPI)
plt.plot(df['t'], df['Uang'], label='Angle Output')
plt.plot(df['t'], df['Uspd'], label='Speed Output')
plt.plot(df['t'], df['Uturn'], label='Turn Output')
plt.title('PID Outputs Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Control Output')
plt.legend()
plt.tight_layout()
plt.savefig('pid_outputs_.png', dpi=DPI, bbox_inches='tight')
plt.close()

# 4. Gyroscope Values vs Time
plt.figure(figsize=FIGSIZE, dpi=DPI)
plt.plot(df['t'], df['gx'], label='Gyro X')
plt.plot(df['t'], df['gy'], label='Gyro Y')
plt.plot(df['t'], df['gz'], label='Gyro Z')
plt.title('Gyroscope Values Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Gyroscope (°/s)')
plt.legend()
plt.tight_layout()
plt.savefig('gyro_values_.png', dpi=DPI, bbox_inches='tight')
plt.close()

# 5. Error Histogram
plt.figure(figsize=FIGSIZE, dpi=DPI)
sns.histplot(df['err'], bins=50, kde=True, color='purple')
plt.title('Error Distribution Histogram')
plt.xlabel('Error (°)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('error_histogram_.png', dpi=DPI, bbox_inches='tight')
plt.close()

# 6. 3D Control Surface: err vs ang vs Uang
fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
ax = fig.add_subplot(111, projection='3d')
sampled = df.sample(n=min(1000, len(df)))  # Reduce points for clarity
ax.scatter(sampled['err'], sampled['ang'], sampled['Uang'],
           c=sampled['Uang'], cmap='viridis', s=5)
ax.set_title('Control Surface: err vs ang vs Uang')
ax.set_xlabel('Error (°)')
ax.set_ylabel('Angle (°)')
ax.set_zlabel('Angle Output (Uang)')
plt.tight_layout()
plt.savefig('control_surface_.png', dpi=DPI, bbox_inches='tight')
plt.close()

# List of files and their corresponding PID values
pid_configs = [
    {"file": "balance_data_2.csv", "Kp": 40.0, "Kd": 1.0, "label": "Kp=30, Kd=1"},
    {"file": "balance_data.csv", "Kp": 50.0, "Kd": 3.0, "label": "Kp=50, Kd=3"},
]

# Create figure with 2 rows (P and D), N columns = number of PID configs
fig, axes = plt.subplots(2, len(pid_configs), figsize=(19.2, 10.8), dpi=200)

# Loop through each config and plot its P and D in corresponding column
for i, config in enumerate(pid_configs):
    df = pd.read_csv(config["file"])

    # Compute PID terms
    P_term = config["Kp"] * df["err"]
    D_term = config["Kd"] * np.gradient(df["err"], df["t"])

    # Smoothing
    window_size = 11 if len(df) >= 11 else (len(df) // 2) * 2 + 1
    P_smooth = savgol_filter(P_term, window_length=window_size, polyorder=2)
    D_smooth = savgol_filter(D_term, window_length=window_size, polyorder=2)

    # Plot Proportional term (top row)
    axes[0, i].plot(df["t"], P_smooth, color='tab:blue', linestyle='--')
    axes[0, i].set_title(f'P Term - {config["label"]}')
    axes[0, i].set_xlabel('Time (s)')
    axes[0, i].set_ylabel('P Contribution')
    axes[0, i].grid(True, linestyle='--', alpha=0.5)

    # Plot Derivative term (bottom row)
    axes[1, i].plot(df["t"], D_smooth, color='tab:red', linestyle='-')
    axes[1, i].set_title(f'D Term - {config["label"]}')
    axes[1, i].set_xlabel('Time (s)')
    axes[1, i].set_ylabel('D Contribution')
    axes[1, i].grid(True, linestyle='--', alpha=0.5)

# Layout adjustments
plt.suptitle('PID Component Comparison by Configuration', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('pid_pd_matrix_comparison_.png', dpi=288, bbox_inches='tight')
plt.close()

