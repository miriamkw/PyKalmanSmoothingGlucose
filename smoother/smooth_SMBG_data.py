import numpy as np
from smoother.closest_values import closest_values
from smoother.convert_to_absolute_time import convert_to_absolute_time
from smoother.convert_to_realtive_time import convert_to_relative_time
from smoother.interpolated_values import interpolated_values
from smoother.set_dynamic_model import set_dynamic_model
from smoother.autodetect_glucose_unit import autodetect_glucose_unit
from smoother.convert_to_mmol_L import convert_to_mmol_L
from smoother.set_iso_error import set_iso_error


def smooth_smbg_data(t_in, y_in, outlier_removal=1, outlier_sd_limit=2, dynamic_model=2):
    """
    Creates a smoothed glucose curve from input glucose readings
    assumed to come from a Self Monitoring Blood Glucose meter.

    Parameters:
    - t_in: array-like, `time in datetime or minutes.`
    - y_in: array-like, glucose values.
    - y_error: [] or array of same length as y.
    - outlier_removal: 0, 1 or 2.
    - outlierSDlimit: number > 0.
    - dynamic_model: 1, 2 or 3.
    - tout: user-supplied vector of times for estimates.
    - unit: string describing glucose units.

    Returns:
    - output: dictionary with smoothed values and statistics.
    """

    # Default parameters
    params = {
        'outlierRemoval': outlier_removal,
        'outlierSDlimit': outlier_sd_limit,
        'dynamicModel': dynamic_model,
        'y_error': [],
        'tout': [],
        'unit': 'auto'
    }

    # Handle unit
    params['unit'] = autodetect_glucose_unit(y_in)
    # This code assumes mmol/L, so we convert it back to mg/dL in the end
    if params['unit'] == 'mg_dL':
        y_in = convert_to_mmol_L(y_in)
        params['y_error'] = convert_to_mmol_L(params['y_error'])

    # Handle time
    start_date_time = t_in[0]
    t_in = convert_to_relative_time(t_in, start_date_time)


    # Set dynamic model
    dyn_model = set_dynamic_model(params['dynamicModel'])
    output = {'delta_t': dyn_model['delta_t']}
    n_states = len(dyn_model['H'])

    # Prepare time for interpolation
    t_i = np.arange(float(t_in[0]), float(t_in[-1]) + 0.1, dyn_model['delta_t'])

    # Filter out NaN values
    valid_mask = ~np.isnan(y_in)
    y = [float(val) for val in y_in[valid_mask]]
    t = [float(val) for val in t_in[valid_mask]]
    t_i_first = np.searchsorted(t_i, t[0], side='right') - 1
    t_i_last = np.searchsorted(t_i, t[-1], side='left')
    if t_i_last < len(t_i):
        t_i_last += 1
    t_i_valid = t_i[t_i_first:t_i_last + 1]

    # Set up error to variance computation
    def error2var(error):
        sdsInConfInterval = 2  # 2 for 95 % CI, 2.5 for 99 %CI
        return (error / sdsInConfInterval) ** 2

    # Set y error
    y_error = set_iso_error(y)
    outliers = np.zeros_like(y, dtype=bool)
    done_finding_outliers = False

    while not done_finding_outliers:
        # Storage
        x_hat_f = np.zeros((n_states, len(t_i_valid)))  # A priori state vector storage, forward pass
        x_bar_f = np.zeros((n_states, len(t_i_valid)))  # A posteriori state vector storage, forward pass
        P_hat_f = np.zeros((n_states, n_states, len(t_i_valid)))  # A priori covariance matrix storage, forward pass
        P_bar_f = np.zeros((n_states, n_states, len(t_i_valid)))  # A posteriori covariance matrix storage, forward pass
        x_smoothed = np.zeros((n_states, len(t_i_valid)))  # State vector storage, backward pass
        P_smoothed = np.zeros((n_states, n_states, len(t_i_valid)))  # Covariance matrix storage, backward pass

        # Initialization
        xBar = np.zeros((n_states, 1))
        xBar[0] = y[0]
        xHat = xBar
        PBar = dyn_model['initCov']
        PHat = PBar
        l = 0  # Adjust index to start from 0 for Python

        # Kalman filter forward pass
        for k in range(len(t_i_valid)):
            # TU - Time update
            xBar = dyn_model['Phi'] @ xHat
            PBar = dyn_model['Phi'] @ PHat @ dyn_model['Phi'].T + dyn_model['Q']

            # Store
            x_bar_f[:, k] = xBar.flatten()
            P_bar_f[:, :, k] = PBar

            meas_update_done = False

            # MU - Measurement Update only when we have a measurement
            while l < len(t) and t_i_valid[k] >= t[l]:  # Interpolated time has passed one of the measurement times
                if meas_update_done:
                    # More than one measurement at the current time
                    xBar = xHat
                    PBar = PHat

                dz = y[l] - dyn_model['H'] @ xBar
                R = error2var(y_error[l])
                Pz = (dyn_model['H'] @ PBar @ dyn_model['H'].T + R)

                if params['outlierRemoval'] == 2:
                    # Check the innovation
                    if abs(dz) > params['outlierSDlimit'] * np.sqrt(Pz):
                        outliers[l] = True
                        print(f"Forward pass flagged measurement as outlier: t = {t[l]}, y = {y[l]} [mmol/L].")

                if not outliers[l]:
                    # Measurement update
                    K = PBar @ dyn_model['H'].T / Pz
                    xHat = (xBar.T + K * dz).T
                    PHat = (np.eye(PBar.shape[0]) - K[:, np.newaxis] * dyn_model['H'][np.newaxis, :]) @ PBar
                    meas_update_done = True
                l += 1

            if not meas_update_done:  # No measurement was available at this time
                xHat = xBar
                PHat = PBar

            # Limit to strictly positive for those states that are strictly positive
            xHat[dyn_model['strictlyPositiveStates'] & (xHat < 0)] = 0

            # Store
            x_hat_f[:, k] = xHat.flatten()
            P_hat_f[:, :, k] = PHat

        # Rauch-Tung-Striebel backward pass
        k = len(t_i_valid) - 1
        x_smoothed[:, k] = xHat.flatten()
        P_smoothed[:, :, k] = PHat

        for k in range(len(t_i_valid) - 2, -1, -1):  # in Python, range is 0-indexed, so -2:0 gives the same behavior
            C = np.dot(P_hat_f[:, :, k], dyn_model['Phi'].T) @ np.linalg.inv(P_bar_f[:, :, k + 1])
            x_smoothed[:, k] = x_hat_f[:, k] + np.dot(C, (x_smoothed[:, k + 1] - x_bar_f[:, k + 1]))
            P_smoothed[:, :, k] = P_hat_f[:, :, k] + np.dot(C, np.dot((P_smoothed[:, :, k + 1] - P_bar_f[:, :, k + 1]),
                                                                      C.T))
            strictly_positive_mask = dyn_model['strictlyPositiveStates'][:, 0]
            x_smoothed[strictly_positive_mask & (x_smoothed[:, k] < 0), k] = 0

        # Generate output struct
        output = {
            'y_smoothed': np.full_like(t_i, np.nan),
            'y_smoothed_sd': np.full_like(t_i, np.nan)
        }
        output['y_smoothed'][t_i_first:t_i_last+1] = x_smoothed[0, :]

        for k in range(len(t_i_valid)):
            output['y_smoothed_sd'][t_i_first - 1 + k] = np.sqrt(P_smoothed[0, 0, k])

        if params['outlierRemoval'] == 1:
            # Run through all measurements and see if any are outside the error smoothed band, if so they are outliers
            found_new_outliers = False
            y_s_mean = closest_values(t, t_i, output['y_smoothed'], start_date_time)
            y_s_sd = closest_values(t, t_i, output['y_smoothed_sd'], start_date_time)
            for i in range(len(y)):
                if not outliers[i] and abs(y[i] - y_s_mean[i]) / y_s_sd[i] > params['outlierSDlimit']:
                    outliers[i] = True
                    found_new_outliers = True
                    print(f"Smoother flagged measurement {i} as outlier: t = {t[i]}, y = {y[i]} [mmol/L].")

            if not found_new_outliers:
                done_finding_outliers = True
            else:
                print(
                    f"Smoother needs a second pass due to outliers detected. Total # outliers in input data: {np.sum(outliers)}")
        else:
            done_finding_outliers = True

    # Smoothing done
    t_in = [float(val) for val in t_in]
    output['outliers'] = closest_values(t_in, t, outliers, start_date_time) == 1
    output['y_filtered'] = np.full_like(t_i, np.nan)
    output['y_filtered_sd'] = np.full_like(t_i, np.nan)
    output['y_filtered'][t_i_first:t_i_last + 1] = x_hat_f[0, :]
    for k in range(len(t_i_valid)):
        output['y_filtered_sd'][t_i_first - 1 + k] = np.sqrt(P_hat_f[0, 0, k])
    output['t_i'] = t_i
    if isinstance(start_date_time, np.datetime64):  # Assuming you have imported datetime
        output['t_i_relative'] = t_i
        output['t_i'] = convert_to_absolute_time(t_i, start_date_time)

    # Add internal states
    output['x_filtered'] = np.full((x_hat_f.shape[0], t_i.shape[0]), np.nan)
    output['x_smoothed'] = np.full((x_smoothed.shape[0], t_i.shape[0]), np.nan)
    output['x_filtered'][:, t_i_first:t_i_last+1] = x_hat_f
    output['x_smoothed'][:, t_i_first:t_i_last+1] = x_smoothed

    # Add user supplied wanted times
    if len(params['tout']) == 0:
        params['tout'] = t_in
    output['y_smoothed_at_tout'] = interpolated_values(params['tout'], t_i, output['y_smoothed'], start_date_time)
    output['y_smoothed_sd_at_tout'] = interpolated_values(params['tout'], t_i, output['y_smoothed_sd'],
                                                         start_date_time)

    # Add dynModel
    output['dynModel'] = dyn_model

    return output
