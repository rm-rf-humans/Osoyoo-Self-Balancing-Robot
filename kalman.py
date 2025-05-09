import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

fs = 100  
dt = 1 / fs

# Load your data
df = pd.read_csv('kalman.csv')  # Replace with your filename
n_samples = len(df)
df['t'] = np.linspace(0, (n_samples - 1) * dt, n_samples)  # Create synthetic time column

axes = ['x', 'y', 'z']

for axis in axes:
    raw_col = f'g{axis}_raw'
    filt_col = f'g{axis}'

    # Time-series Plot
    plt.figure(figsize=(13.33, 7.5), dpi=288)
    plt.plot(df['t'], df[raw_col], label=f'Raw Gyro {axis.upper()}', alpha=0.5, color='gray')
    plt.plot(df['t'], df[filt_col], label=f'Filtered Gyro {axis.upper()}', color='tab:blue')
    plt.title(f'Raw vs Kalman-Filtered Gyro ({axis.upper()} Axis)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (deg/s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'gyro_raw_vs_filtered_{axis}_.png', dpi=288)
    plt.close()

    # Noise stats
    raw_std = df[raw_col].std()
    filt_std = df[filt_col].std()
    reduction = (1 - filt_std / raw_std) * 100
    print(f"[{axis.upper()}] Noise Reduction: {reduction:.2f}%")

    # PSD
    f_raw, Pxx_raw = welch(df[raw_col], fs=fs)
    f_filt, Pxx_filt = welch(df[filt_col], fs=fs)
    plt.figure(figsize=(13.33, 7.5), dpi=288)
    plt.semilogy(f_raw, Pxx_raw, label=f'Raw Gyro {axis.upper()}')
    plt.semilogy(f_filt, Pxx_filt, label=f'Filtered Gyro {axis.upper()}')
    plt.title(f'PSD of Gyro Signals ({axis.upper()} Axis)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'gyro_psd_comparison_{axis}_.png', dpi=288)
    plt.close()

    # Integrated Angle Drift
    angle_raw = np.cumsum(df[raw_col] * dt)
    angle_filt = np.cumsum(df[filt_col] * dt)
    plt.figure(figsize=(13.33, 7.5), dpi=288)
    plt.plot(df['t'], angle_raw, label=f'Raw Integrated {axis.upper()}', color='gray')
    plt.plot(df['t'], angle_filt, label=f'Filtered Integrated {axis.upper()}', color='tab:blue')
    plt.title(f'Gyro Drift (Integrated Angle) - {axis.upper()} Axis')
    plt.xlabel('Time (s)')
    plt.ylabel('Estimated Angle (deg)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'gyro_drift_comparison_{axis}_.png', dpi=288)
    plt.close()

