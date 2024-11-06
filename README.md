# PyKalmanSmoothingGlucose

[![MATLAB Implementation](https://img.shields.io/badge/MATLAB-Implementation-brightgreen)](https://github.com/omstaal/kalman-smoothing-glucose?tab=readme-ov-file)

This is a Python implementation of the Kalman smoothing described in [1]. Matlab implementation is available [here](https://github.com/omstaal/kalman-smoothing-glucose?tab=readme-ov-file). The filter can apply (offline) Kalman smoothing, perform outlier detection and removal, and estimate measurement uncertainty.

![FIGURE NOT AVAILABLE](figures/kalman_smoothing.png "Kalman smoothing example output.")

## Getting Started

Setup and activate virtual environment with `python -m venv kalman_venv`. Activate it with (Mac) `source kalman_venv/bin/activate` or (Windows) `.kalman_venv\Scripts\activate` 

Install requirements:
```
pip install -r requirements.txt
```
Run test example:
```
python example/test.py
```

## Description of Usage

### Main Function Signature 
```
def smooth_smbg_data(t_in, y_in, outlier_removal=1, dynamic_model=2):
```
Parameters:
- `t_in:` 
A list or array of time points corresponding to the SMBG data, in datetime or minutes.
- `y_in:`
A list or array of glucose data values that correspond to the time points in t_in. Should be of the same length as t_in.

- `outlier_removal (int, optional):`
- A flag to enable/disable outlier removal. Default is 1 (enabled). Possible values:
  - 0: Disable outlier removal 
  - 1: Use a fixed threshold based on standard deviation 
  - 2: Analyze innovation using standard deviation and covariance (better for dynamic/noisy data)
- `outlier_sd_limit`: A positive number (>0) that sets the threshold for detecting outliers. The default is 2. This value is used to multiply the standard deviation to determine the outlier threshold and is therefore dimensionless.
- `dynamic_model (int, optional):`
  An identifier for the dynamic model to use in the smoothing process. Default is 2.  
  Possible inputs:
  - **0**: No dynamics (static system).
  - **1**: Simple 2nd-order system.
  - **2**: Lumped insulin/meal state (central/remote).
  - **3**: Insulin and meal, central.

  For a detailed description of each model, refer to the file [`set_dynamic_model.py`](smoother/set_dynamic_model.py).

### Example Use

The main function is called smooth_smbg_data. 
```
from smoother.smooth_SMBG_data import smooth_smbg_data

# Load the data
y = data['Glucose'].values
t = data['DateTime'].values

# Smooth the data (replace with equivalent smoothing function)
smoother_result = smooth_smbg_data(t, y)
```

### Main Function Output Dictionary

The output dictionary contains the smoothed SMBG data along with additional information, such as outliers, filtered values, and internal states. Here's a breakdown of the keys and their respective values:

- **`delta_t`**: The time step used in the dynamic model.
- **`y_smoothed`**: An array of the smoothed glucose measurements corresponding to the input timestamps.
- **`y_smoothed_sd`**: The standard deviation of the smoothed glucose measurements, representing the uncertainty in the smoothed values.
- **`outliers`**: A boolean array that flags outliers based on the selected outlier detection method. It marks whether each data point is considered an outlier or not.
- **`y_filtered`**: An array of filtered glucose values after applying the dynamic model and outlier removal.
- **`y_filtered_sd`**: The standard deviation of the filtered glucose values, representing the uncertainty in the filtered data.
- **`t_i`**: The time indices corresponding to the filtered and smoothed data.
- **`t_i_relative`**: This contains the relative time indices (`t_i`) with respect to the start time, in minutes.
- **`x_filtered`**: An array representing the filtered internal state estimates from the dynamic model.
- **`x_smoothed`**: An array representing the smoothed internal state estimates, similar to `x_filtered`, but with smoothed values.
- **`y_smoothed_at_tout`**: An array of the smoothed glucose values interpolated to the user-supplied `tout` times.
- **`y_smoothed_sd_at_tout`**: An array of the smoothed standard deviations, interpolated to the user-supplied `tout` times.
- **`dynModel`**: The dynamic model used for smoothing, represented as a dictionary.


## References
[1] [Staal, O. M., Sælid, S., Fougner, A., & Stavdahl, Ø. (Year). *Kalman Smoothing for Objective and Automatic Preprocessing of Glucose Data*. IEEE Journal of Biomedical and Health Informatics.](https://ieeexplore.ieee.org/document/8305603)


