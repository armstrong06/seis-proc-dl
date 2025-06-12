import h5py
import numpy as np

with h5py.File("YSnoiseZ_4s_1ex.h5", "r") as f:
    ex = f["X"][0, :, :]

assert ex.shape == (400, 1)

norm_vals = np.max(abs(ex), axis=0)
assert norm_vals.shape[0] == 1

ex_norm = ex / norm_vals

assert np.max(abs(ex_norm)) == 1

assert ex_norm.shape == (400, 1)
ex_norm = ex_norm[None, :, :]
assert ex_norm.shape == (1, 400, 1)

with h5py.File("YSnoiseZ_4s_1ex_norm.h5", "w") as f:
    f.create_dataset("X", data=ex_norm)
