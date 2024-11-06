import numpy as np
import pandas as pd


def closest_values(tout, t, y, startdatetime=None):
    """
    Helper function to find closest values to wanted output times.

    Parameters:
    tout (array-like): The desired output times.
    t (array-like): The time series of the original data.
    y (array-like): The values corresponding to the time series.
    startdatetime (datetime, optional): The starting datetime for relative time conversion.

    Returns:
    np.ndarray: The closest values corresponding to tout.
    """
    # Convert tout to relative time if startdatetime is provided
    if startdatetime is not None and isinstance(tout, (pd.Series, pd.DatetimeIndex)):
        # TODO:
        print("IS CALLED???? DELETE IF NEVER PROMPTED!")
        tout = (tout - startdatetime).total_seconds()  # Convert to seconds or any relative time scale

    # Check if tout is monotonously increasing
    if np.any(np.diff(tout) < 0):
        raise ValueError('tout was not monotonously increasing')

    yout = np.zeros(len(tout))  # Initialize output array
    lastj = 0  # Initialize the index for t

    for i in range(len(tout)):
        if tout[i] > t[-1] or tout[i] < t[0]:
            yout[i] = np.nan  # Assign NaN if out of bounds
        else:
            # Find the closest value in t
            for j in range(lastj, len(t)):
                if t[j] >= tout[i]:
                    yout[i] = y[j]
                    lastj = j  # Update lastj to current index
                    break

    return yout

