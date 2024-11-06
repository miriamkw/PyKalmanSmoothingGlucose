import numpy as np
from scipy.linalg import expm


def set_dynamic_model(dynModelNo):
    dynModel = {}
    dynModel['delta_t'] = 1 / 6  # 1 min / 6 = 10 sec

    if dynModelNo == 0:  # No dynamics
        dynModel['id'] = 0
        dynModel['F'] = np.array([[0]])
        dynModel['Q'] = np.array([[0.005 * dynModel['delta_t']]])
        dynModel['H'] = np.array([[1]])
        dynModel['initCov'] = np.diag([0.25])
        dynModel['Phi'] = expm(dynModel['F'] * dynModel['delta_t'])
        dynModel['stateNames'] = ['Gp']
        dynModel['strictlyPositiveStates'] = [True]

    elif dynModelNo == 1:  # Simple 2nd order system
        dynModel['id'] = 1
        a = -0.05
        qm1 = 0.005 * dynModel['delta_t']
        dynModel['F'] = np.array([[0, 1], [0, a]])
        dynModel['Q'] = np.array([[0, 0], [0, qm1]])
        dynModel['H'] = np.array([1, 0])
        dynModel['initCov'] = np.diag([0.25, 1])
        dynModel['Phi'] = expm(dynModel['F'] * dynModel['delta_t'])
        dynModel['stateNames'] = ['Gp', 'dGp']
        dynModel['strictlyPositiveStates'] = [True, False]

    elif dynModelNo == 2:  # Lumped insulin/meal state (central/remote)
        dynModel['id'] = 2
        Td = 10.0
        qm2 = 0.02 * dynModel['delta_t']
        dynModel['F'] = np.array([[0, 0, 1], [0, -1 / Td, 0], [0, 1 / Td, -1 / Td]])
        dynModel['Q'] = np.array([[0, 0, 0], [0, qm2, 0], [0, 0, 0]])
        dynModel['H'] = np.array([1, 0, 0])
        dynModel['initCov'] = np.diag([10, 1, 1])
        dynModel['Phi'] = expm(dynModel['F'] * dynModel['delta_t'])
        dynModel['stateNames'] = ['Gp', 'C', 'R']
        dynModel['strictlyPositiveStates'] = np.array([[True], [False], [False]])

    elif dynModelNo == 3:  # Insulin and meal, central
        dynModel['id'] = 3
        Ti = 20.0
        Tm = 10.0
        qm3i = 0.01 * dynModel['delta_t']
        qm3m = 0.01 * dynModel['delta_t']
        dynModel['F'] = np.array([[0, -1, 1], [0, -1 / Ti, 0], [0, 0, -1 / Tm]])
        dynModel['Q'] = np.array([[0, 0, 0], [0, qm3i, 0], [0, 0, qm3m]])
        dynModel['H'] = np.array([1, 0, 0])
        dynModel['initCov'] = np.diag([10, 1, 1])
        dynModel['Phi'] = expm(dynModel['F'] * dynModel['delta_t'])
        dynModel['stateNames'] = ['Gp', 'I', 'M']
        dynModel['strictlyPositiveStates'] = [True, True, True]
    else:
        raise ValueError(f"Unsupported model: {dynModelNo}")

    return dynModel

