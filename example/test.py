import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Adjust to test for mg/dL
use_mg_dl = True

# Adjust the path to point to the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from smoother.smooth_SMBG_data import smooth_smbg_data

# Load the data
test_file_path = os.path.join('example', 'test_data.txt')
data = pd.read_csv(test_file_path, delimiter='\t')  # Adjust delimiter if needed
data['DateTime'] = pd.to_datetime(data['DateTime'], dayfirst=True)  # Ensure DateTime format
data['Glucose'] = data['Glucose'].astype(float)

# Extract variables
y = data['Glucose'].values
if use_mg_dl:
    y = np.array([val * 18 for val in y])
t = data['DateTime'].values

# Smooth the data (replace with equivalent smoothing function)
# Example: Savitzky-Golay filter, adapt parameters to your needs
smoother_result = smooth_smbg_data(t, y)

# Save the smoothed table back to file
# Uncomment to save
# output.to_csv('test_smoothed.txt', index=False)

# Plotting
draw_plot = True
if draw_plot:
    plt.figure()

    # Plot the smoothed glucose values as a blue line with width 2
    plt.plot(smoother_result['t_i'], smoother_result['y_smoothed'], 'b-', linewidth=2)

    # Plot the 95% confidence interval as blue dashed lines
    plt.plot(smoother_result['t_i'], smoother_result['y_smoothed'] + 2 * smoother_result['y_smoothed_sd'], 'b--')

    # Plot resampled values every five minutes with circles
    resample_interval = 6 * 5  # 5 minutes, smoothing sampling rate is 10 seconds times per minute
    resample_indices = np.arange(0, len(smoother_result['t_i']), resample_interval)
    plt.plot(smoother_result['t_i'][resample_indices], smoother_result['y_smoothed'][resample_indices],
             'ko')

    plt.plot(t, y, 'r.', markersize=10)

    # Plot the outliers with black crosses ('kx') and marker size 10
    ol = smoother_result['outliers'] == 1
    plt.plot(t[ol], y[ol], 'kx', markersize=13)

    plt.plot(smoother_result['t_i'], smoother_result['y_smoothed'] - 2 * smoother_result['y_smoothed_sd'], 'b--')

    plt.legend(['Smoothed glucose', '95% CI of estimate', 'Resampled measurements', 'Input glucose measurements', 'Outliers'], loc='upper left')
    plt.show()


