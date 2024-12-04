import numpy as np

def convert_to_mg_dl(y_mmol_L):
    return np.array([val * 18.018 for val in y_mmol_L])

