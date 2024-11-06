import numpy as np
import pandas as pd

from smoother.autodetect_glucose_unit import autodetect_glucose_unit
from smoother.convert_to_realtive_time import convert_to_relative_time


def smooth_smbg_and_cgm_data(t_in, y_in_fp, y_in_cgm, *args):
    # Parse variable arguments
    parsed_args = parse_input_varargs(args)

    # Check units
    unit1 = autodetect_glucose_unit(y_in_fp)
    unit2 = autodetect_glucose_unit(y_in_cgm)
    if unit1 != unit2:
        raise ValueError('Supplied units are not consistent for fingerpricks and CGM')

    if unit1 != 'mmol_L':
        raise NotImplementedError('Unsupported unit')

    # Handle time
    start_datetime = None
    if isinstance(t_in, pd.DatetimeIndex):
        start_datetime = t_in[0]
        t_in = convert_to_relative_time(t_in, start_datetime)

    # Run smoothing on each set individually
    smoothed_fp = smooth_smbg_data(t_in, y_in_fp, outlier_removal=parsed_args['outlier_removal'],
                                   dynamic_model=parsed_args['dynamic_model'])
    smoothed_cgm = smooth_smbg_data(t_in, y_in_cgm, outlier_removal=parsed_args['outlier_removal'],
                                    dynamic_model=parsed_args['dynamic_model'])

    t_i = smoothed_fp['t_i']
    y_fp = smoothed_fp['y_smoothed']
    var_fp = smoothed_fp['y_smoothed_sd'] ** 2
    y_cgm = smoothed_cgm['y_smoothed']
    var_cgm = smoothed_cgm['y_smoothed_sd'] ** 2

    # Set dynamic model to use for CGM SMBG fusion
    dyn_model = augment_dynamic_model_bias_and_lag(smoothed_fp['dyn_model'])
    n_states = dyn_model['Phi'].shape[0]

    # Storage
    x_hat_f = np.full((n_states, len(t_i)), np.nan)  # A priori state vector storage, forward pass
    x_bar_f = np.full((n_states, len(t_i)), np.nan)  # A posteriori state vector storage, forward pass
    p_hat_f = np.full((n_states, n_states, len(t_i)), np.nan)  # A priori covariance matrix storage, forward pass
    p_bar_f = np.full((n_states, n_states, len(t_i)), np.nan)  # A posteriori covariance matrix storage, forward pass
    x_smoothed = np.full((n_states, len(t_i)), np.nan)  # State vector storage, backward pass
    p_smoothed = np.full((n_states, n_states, len(t_i)), np.nan)  # Covariance matrix storage, backward pass
    phis = np.full((n_states, n_states, len(t_i)), np.nan)
    phi_ekfs = np.full((n_states, n_states, len(t_i)), np.nan)

    mink = 1 / 30
    maxk = 10

    init = False
    endk = np.nan
    startk = 0

    for k in range(len(t_i)):
        if not init:
            if not np.isnan(y_fp[k]) and not np.isnan(y_cgm[k]):  # Initialization
                # Initialization
                x_bar = np.zeros(n_states)
                x_bar[0] = y_fp[k]
                x_bar[dyn_model['Nin']] = y_fp[k]  # Set the state equal to the first CGM measurement, due to bias
                bias_start_guess = parsed_args['bias_start_guess']

                if bias_start_guess == 0:
                    x_bar[dyn_model['Nin'] + 1] = 0
                    dyn_model['init_cov'][dyn_model['Nin'] + 1, dyn_model['Nin'] + 1] = 1  # If we do not use the data
                elif bias_start_guess == 1:
                    x_bar[dyn_model['Nin'] + 1] = y_cgm[k] - y_fp[
                        k]  # Start guess on the bias - first sample difference
                elif bias_start_guess == 2:
                    x_bar[dyn_model['Nin'] + 1] = np.nanmean(y_cgm - y_fp)  # Start guess on the bias - mean difference
                elif bias_start_guess == 3:
                    x_bar[dyn_model['Nin'] + 1] = (y_cgm[k] - y_fp[k] + np.nanmean(y_cgm - y_fp)) / 2  # Start guess
                print(f'Used guess for bias: {x_bar[dyn_model["Nin"] + 1]}')  # Start guess on the bias

                # The guess needs to be computed in a different way if fingerpricks are sparse
                x_bar[dyn_model['Nin'] + 2] = dyn_model['k_default']

                x_hat = x_bar
                p_bar = dyn_model['init_cov']
                p_hat = p_bar
                init = True
                startk = k
        elif init and (np.isnan(y_fp[k]) or np.isnan(y_cgm[k])):  # Time to end
            endk = k - 1
            break
        else:  # We have initialized and it is not time to end yet, do filtering
            # Kalman filter forward pass
            if dyn_model['non_linear']:
                phi, phi_ekf = compute_phi_ekf(dyn_model, x_hat)
                phis[:, :, k] = phi
                phi_ekfs[:, :, k] = phi_ekf

            x_bar = dyn_model['Phi'].dot(x_hat)
            p_bar = phi_ekf.dot(p_hat).dot(phi_ekf.T) + dyn_model['Q']
            x_bar_f[:, k] = x_bar
            p_bar_f[:, :, k] = p_bar

            # Measurement Update
            if not np.isnan(y_fp[k]) and not np.isnan(y_cgm[k]):
                H = dyn_model['H_both']
                y = np.array([y_fp[k], y_cgm[k]])
                R = np.array([[var_fp[k], 0], [0, var_cgm[k]]])
            elif not np.isnan(y_cgm[k]):
                H = dyn_model['H_cgm']
                y = y_cgm[k]
                R = var_cgm[k]
            elif not np.isnan(y_fp[k]):
                H = dyn_model['H_fp']
                y = y_fp[k]
                R = var_fp[k]
            else:
                raise ValueError('Encountered empty measurement, should have been filtered out, something is wrong')

            dz = y - H.dot(x_bar)
            Pz = H.dot(p_bar).dot(H.T) + R

            # Measurement update
            K = p_bar.dot(H.T) / Pz
            x_hat = x_bar + K.dot(dz)
            hlp = np.eye(p_bar.shape[0]) - K.dot(H)
            p_hat = hlp.dot(p_bar).dot(hlp.T) + K.dot(R).dot(K.T)

            # Check for positive definiteness
            if np.linalg.matrix_rank(p_hat) < p_hat.shape[0]:
                raise ValueError(f'p_hat not positive definite at k={k}')

            if not np.allclose(p_hat, p_hat.T, atol=1e-10):
                p_hat = (p_hat + p_hat.T) / 2  # Make symmetric

            # Clip state
            if x_hat[-1] > maxk:
                x_hat[-1] = maxk
            if x_hat[-1] < mink:
                x_hat[-1] = mink

            # Store
            x_hat_f[:, k] = x_hat
            p_hat_f[:, :, k] = p_hat

    if np.isnan(endk):
        endk = len(t_i)

    # Rauch-Tung-Striebel backward pass
    x_smoothed[:, endk] = x_hat
    p_smoothed[:, :, endk] = p_hat
    aborted_smoothing = False
    for k in range(endk - 1, startk, -1):
        if dyn_model['non_linear']:
            phi_for_rts = phi_ekfs[:, :, k]

        C = (p_hat_f[:, :, k].dot(phi_for_rts.T)) / p_bar_f[:, :, k + 1]
        x_smoothed[:, k] = x_hat_f[:, k] + C.dot(x_smoothed[:, k + 1] - x_bar_f[:, k + 1])

        # Clip smoothed state
        if x_smoothed[-1, k] > maxk:
            x_smoothed[-1, k] = maxk
        if x_smoothed[-1, k] < mink:
            x_smoothed[-1, k] = mink

        # Covariance matrix
        p_smoothed[:, :, k] = p_hat_f[:, :, k] + C.dot(p_smoothed[:, :, k + 1] - p_bar_f[:, :, k + 1]).dot(C.T)

    # Output
    output = {
        't_i': t_i,
        'y_smoothed': x_smoothed[0, :],
        'y_smoothed_sd': np.sqrt(np.diagonal(p_smoothed[0, 0, :])),
        'y_smoothed2': x_smoothed[1, :],
        'y_smoothed2_sd': np.sqrt(np.diagonal(p_smoothed[1, 1, :])),
        'dyn_model': dyn_model,
        'x_hat_f': x_hat_f,
        'p_hat_f': p_hat_f,
        'phis': phis,
        'phi_ekfs': phi_ekfs
    }

    return output
