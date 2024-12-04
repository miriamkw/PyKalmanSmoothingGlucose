import numpy as np

def autodetect_glucose_unit(measurements):
    unit = 'mmol_L'

    if np.nanmean(measurements) > 50:
        print('Autodetected mg/dL as unit')
        unit = 'mg_dL'
    return unit

