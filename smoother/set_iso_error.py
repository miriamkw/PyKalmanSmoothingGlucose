import numpy as np

def set_iso_error(y):
    """
    SETISOERROR Helper method that guesses an error based on the measured value, based on ISO 15197.
    Assumes y is given in mmol/L.

    Parameters:
    y (np.ndarray): Array of measured values.

    Returns:
    np.ndarray: Array of estimated errors corresponding to the measured values.
    """
    y_error = np.zeros_like(y)  # Initialize error array with the same shape as y

    for i in range(len(y)):  # Iterate through each measured value
        if y[i] > 5.55:
            y_error[i] = 0.15 * y[i]  # Calculate error for values above 5.55 mmol/L
        else:
            y_error[i] = 0.83  # Constant error for values 5.55 mmol/L and below

    return y_error
