import numpy as np

def convert_to_mmol_L(y_mg_dl):
    return np.array([val / 18.018 for val in y_mg_dl])


