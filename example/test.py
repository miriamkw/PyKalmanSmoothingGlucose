import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

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
    plt.plot(t, y, 'r.', markersize=15)

    # Plot the outliers with black crosses ('kx') and marker size 10
    ol = smoother_result['outliers'] == 1
    plt.plot(t[ol], y[ol], 'kx', markersize=10)

    # Plot the smoothed glucose values as a blue line with width 2
    plt.plot(smoother_result['t_i'], smoother_result['y_smoothed'], 'b-', linewidth=2)

    # Plot the 95% confidence interval as blue dashed lines
    plt.plot(smoother_result['t_i'], smoother_result['y_smoothed'] + 2 * smoother_result['y_smoothed_sd'], 'b--')
    plt.plot(smoother_result['t_i'], smoother_result['y_smoothed'] - 2 * smoother_result['y_smoothed_sd'], 'b--')

    plt.legend(['Input glucose measurements', 'Outliers', 'Smoothed glucose', '95% CI of estimate'], loc='upper left')
    plt.show()


