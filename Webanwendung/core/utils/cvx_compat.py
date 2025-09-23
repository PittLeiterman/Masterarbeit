# utils/cvx_compat.py
class Const:
    def __init__(self, val):
        import numpy as np
        self.value = np.asarray(val)
