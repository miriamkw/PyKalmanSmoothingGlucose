import numpy as np
import pandas as pd
from smoother.convert_to_realtive_time import convert_to_relative_time


def interpolated_values(tout, t, y, startdatetime=None):
    """
    Helper function to find values at interpolated output times.

    Parameters:
    tout (array-like): The desired output times.
    t (array-like): The time series of the original data.
    y (array-like): The values corresponding to the time series.
    startdatetime (datetime, optional): The starting datetime for relative time conversion.

    Returns:
    np.ndarray: The interpolated values corresponding to tout.
    """
    # Convert tout to relative time if startdatetime is provided
    if startdatetime is not None and isinstance(tout, (pd.Series, pd.DatetimeIndex)):
        tout = convert_to_relative_time(tout, startdatetime)

    # Check if tout is monotonously increasing
    if np.any(np.diff(tout) < 0):
        raise ValueError('tout was not monotonously increasing')

    yout = np.zeros(len(tout))  # Initialize output array
    lastj = 0  # Initialize the index for t

    for i in range(len(tout)):
        if tout[i] > t[-1] or tout[i] < t[0]:
            # Do not extrapolate
            yout[i] = np.nan
        else:
            # Find the closest value in t and interpolate
            for j in range(lastj, len(t)):
                if t[j] == tout[i]:
                    yout[i] = y[j]
                    lastj = j
                    break
                elif t[j] > tout[i]:
                    y1 = y[j - 1]
                    y2 = y[j]
                    t1 = t[j - 1]
                    t2 = t[j]

                    # Perform linear interpolation
                    yout[i] = y1 + (tout[i] - t1) * (y2 - y1) / (t2 - t1)

                    lastj = j
                    break

    return yout
