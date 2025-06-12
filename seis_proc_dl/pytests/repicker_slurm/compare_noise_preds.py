# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
ref_preds1 = np.load("../example_files/repicker_test_exs/seed1_YSnoiseZ_preds_ex1.npy")
ref_preds2 = np.load("../example_files/repicker_test_exs/seed2_YSnoiseZ_preds_ex1.npy")
ref_preds3 = np.load("../example_files/repicker_test_exs/seed3_YSnoiseZ_preds_ex1.npy")
# %%
spdl_preds = np.load("YSnoiseZ_4s_1ex_spdl_preds.npy")
spdl_norm_preds = np.load("YSnoiseZ_4s_1ex_norm_spdl_preds.npy")
spdl_db_preds = np.load("YSnoiseZ_4s_1ex_spdl_db_preds.npy")
comb_refs = np.concatenate([ref_preds1, ref_preds2, ref_preds3], axis=1)

# %%
plt.hist(comb_refs[0, :], bins=np.arange(-0.75, 0.80, 0.05))

# %%
plt.hist(spdl_preds[0, :], bins=np.arange(-0.75, 0.80, 0.05))
# %%
plt.hist(spdl_db_preds[0, :], bins=np.arange(-0.75, 0.80, 0.05))
# %%
plt.hist(spdl_norm_preds[0, :], bins=np.arange(-0.75, 0.80, 0.05))
# %%
print(np.std(comb_refs))
print(np.std(spdl_preds))
print(np.std(spdl_db_preds))
print(np.std(spdl_norm_preds))
# %%
print(np.sum(spdl_preds[0, :] == 0.75))
print(np.sum(spdl_preds[0, :] == -0.75))
# %%
print(np.sum(spdl_db_preds[0, :] == 0.75))
print(np.sum(spdl_db_preds[0, :] == -0.75))

# %%
assert np.allclose(spdl_db_preds, spdl_preds)
# %%
import h5py

with h5py.File("../example_files/repicker_test_exs/YSnoiseZ_4s_1ex.h5") as f:
    X = f["X"][0, :, 0]

plt.plot(np.arange(0, 4, 0.01), X)
# %%
with h5py.File("../example_files/repicker_test_exs/YSnoiseZ_4s_1ex_norm.h5") as f:
    X = f["X"][0, :, 0]

plt.plot(np.arange(0, 4, 0.01), X)
# %%
