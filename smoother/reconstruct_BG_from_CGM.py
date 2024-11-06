import numpy as np
import pandas as pd


def ReconstructBGFromCGM(t_in, y_in_cgm, bias, lag, **kwargs):
    """
    Creates a smoothed SMBG blood glucose curve from CGM glucose readings,
    and an estimated bias and lag constant.

    Parameters:
        t_in : array-like
            Array of datetime or time in minutes as floats.
        y_in_cgm : array-like
            CGM glucose values.
        bias : float
            Estimated bias constant.
        lag : float
            Estimated lag constant.
        kwargs : keyword arguments
            Additional parameters for processing (see MATLAB comments for details).

    Returns:
        dict : Contains smoothed glucose data and other relevant outputs.
    """

    # Parse variable arguments
    parsed_args = parse_input_varargs(kwargs)

    # Check units
    unit = auto_detect_glucose_unit(y_in_cgm)
    if unit != 'mmol/L':
        raise ValueError('Unsupported unit')

    # Handle time
    if isinstance(t_in, pd.DatetimeIndex):
        # Convert to relative time
        start_date_time = t_in[0]
        t_in = convert_to_relative_time(t_in, start_date_time)
    else:
        # Assume relative time is passed in
        start_date_time = None

    # Run smoothing on each set individually
    output = {}
    output['smoothed_cgm'] = SmoothSMBGData(t_in, y_in_cgm, outlierRemoval=parsed_args['outlierRemoval'],
                                            dynamicModel=parsed_args['dynamicModel'])
    t_i = output['smoothed_cgm']['t_i']
    output['t_i'] = t_i
    y_cgm = output['smoothed_cgm']['y_smoothed']
    var_cgm = output['smoothed_cgm']['y_smoothed_sd'] ** 2

    # Set dynamic model to use for CGM SMBG fusion
    dyn_model = augment_dynamic_model_known_lag(output['smoothed_cgm']['dynModel'], lag)
    n_states = dyn_model['Phi'].shape[0]

    # Storage for Kalman filter
    x_hat_f = np.full((n_states, len(t_i)), np.nan)
    x_bar_f = np.full((n_states, len(t_i)), np.nan)
    P_hat_f = np.full((n_states, n_states, len(t_i)), np.nan)
    P_bar_f = np.full((n_states, n_states, len(t_i)), np.nan)
    x_smoothed = np.full((n_states, len(t_i)), np.nan)
    P_smoothed = np.full((n_states, n_states, len(t_i)), np.nan)
    Phis = np.full((n_states, n_states, len(t_i)), np.nan)
    PhiEKFs = np.full((n_states, n_states, len(t_i)), np.nan)

    init = False
    endk = np.nan
    for k in range(len(t_i)):
        if not init:
            if not np.isnan(y_cgm[k]):  # We can initialize
                # Initialization
                x_bar = np.zeros(n_states)
                x_bar[0] = y_cgm[k] - bias
                x_bar[dyn_model['Nin']] = y_cgm[k] - bias

                x_hat = x_bar
                P_bar = dyn_model['initCov']

                PHat = P_bar
                init = True
                startk = k

                H = dyn_model['H']
                R = var_cgm[k]
        elif init and np.isnan(y_cgm[k]):  # Time to end
            endk = k - 1
            break
        else:  # We have initialized and it is not time to end yet, do filtering
            # Kalman filter forward pass
            # TU - Time update
            if dyn_model['nonLinear']:
                dyn_model['Phi'], dyn_model['PhiEKF'] = compute_Phi_EKF(dyn_model, x_hat)
                Phis[:, :, k] = dyn_model['Phi']
                PhiEKFs[:, :, k] = dyn_model['PhiEKF']
            x_bar = dyn_model['Phi'].dot(x_hat)
            P_bar = dyn_model['PhiEKF'].dot(PHat).dot(dyn_model['PhiEKF'].T) + dyn_model['Q']

            # Store
            x_bar_f[:, k] = x_bar
            P_bar_f[:, :, k] = P_bar

            # MU - Measurement Update
            y = y_cgm[k] - bias
            dz = y - H.dot(x_bar)
            Pz = H.dot(P_bar).dot(H.T) + R

            # Measurement update
            K = P_bar.dot(H.T) @ np.linalg.inv(Pz)
            x_hat = x_bar + K.dot(dz)
            hlp = np.eye(PBar.shape[0]) - K.dot(H)
            PHat = hlp.dot(P_bar).dot(hlp.T) + K.dot(R).dot(K.T)

            # Ensure PHat is positive definite
            if not np.all(np.linalg.eigvals(PHat) > 0):
                raise ValueError(f'PHat not positive definite at k={k}')
            if not np.allclose(PHat, PHat.T, atol=1e-8):
                PHat = (PHat + PHat.T) / 2  # Make symmetric

            # Store
            x_hat_f[:, k] = x_hat
            P_hat_f[:, :, k] = PHat

    if np.isnan(endk):
        endk = len(t_i) - 1

    # Rauch-Tung-Striebel backward pass
    x_smoothed[:, endk] = x_hat
    P_smoothed[:, :, endk] = PHat
    output['abortedSmoothing'] = False
    PhiForRTS = dyn_model['Phi']

    for k in range(endk - 1, startk, -1):
        if dyn_model['nonLinear']:
            PhiForRTS = PhiEKFs[:, :, k]
        C = (P_hat_f[:, :, k].dot(PhiForRTS.T)) / P_bar_f[:, :, k + 1]
        x_smoothed[:, k] = x_hat_f[:, k] + C.dot(x_smoothed[:, k + 1] - x_bar_f[:, k + 1])

        P_hat_s = P_hat_f[:, :, k] + C.dot(P_smoothed[:, :, k + 1] - P_bar_f[:, :, k + 1]).dot(C.T)

        # Ensure P_hat_s is positive definite
        if not np.all(np.linalg.eigvals(P_hat_s) > 0):
            print(f'Warning - smoothing aborted at t = {t_i[k]} due to non-posdef covmatrix.')
            output['abortedSmoothing'] = True
            break

        if not np.allclose(P_hat_s, P_hat_s.T, atol=1e-8):
            P_hat_s = (P_hat_s + P_hat_s.T) / 2  # Make symmetric
        P_smoothed[:, :, k] = P_hat_s

    # Generate output structs
    output['y_fprec_smoothed'] = np.full(len(t_i), np.nan)
    output['y_fprec_smoothed_sd'] = np.full(len(t_i), np.nan)
    output['y_cgm_smoothed'] = np.full(len(t_i), np.nan)
    output['y_cgm_smoothed_sd'] = np.full(len(t_i), np.nan)

    output['y_fprec_smoothed'][startk:endk] = x_smoothed[0, startk:endk]
    output['y_cgm_smoothed'][startk:endk] = x_smoothed[dyn_model['Nin'], startk:endk]

    for k in range(startk, endk + 1):
        output['y_fprec_smoothed_sd'][k] = np.sqrt(P_smoothed[0, 0, k])
        output['y_cgm_smoothed_sd'][k] = np.sqrt(P_smoothed[dyn_model['Nin'], dyn_model['Nin'], k])

    # Add internal states
    output['x_filtered'] = x_hat_f
    output['x_smoothed'] = x_smoothed

    # Add user supplied wanted time
    output['y_fprec_at_tout'] = closest_values(parsed_args['tout'], t_i, output['y_fprec_smoothed'], start_date_time)
    output['y_fprec_sd_at_tout'] = closest_values(parsed_args['tout'], t_i, output['y_fprec_smoothed_sd'],
                                                  start_date_time)

    # Debug plotting (optional)
    if parsed_args['debugPlot']:
        sds_in_conf_interval = 2
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t_in, y_in_cgm, 'b.', label='FGM measurements', markersize=15)
        plt.plot(t_i, y_cgm, 'b-', label='Smoothed FGM', linewidth=3)
        plt.fill_between(t_i,
                         output['y_cgm_smoothed'] - sds_in_conf_interval * output['y_cgm_smoothed_sd'],
                         output['y_cgm_smoothed'] + sds_in_conf_interval * output['y_cgm_smoothed_sd'],
                         color='cyan', alpha=0.3, label='2 sd conf. interval')
        plt.xlabel('Time')
        plt.ylabel('Glucose level')
        plt.title('CGM Smoothing Result')
        plt.legend()
        plt.grid()
        plt.show()

    return output

