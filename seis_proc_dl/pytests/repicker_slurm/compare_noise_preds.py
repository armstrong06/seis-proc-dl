# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
ref_preds1 = np.load("../example_files/repicker_test_exs/seed1_YSnoiseZ_preds_ex1.npy")
ref_preds2 = np.load("../example_files/repicker_test_exs/seed2_YSnoiseZ_preds_ex1.npy")
ref_preds3 = np.load("../example_files/repicker_test_exs/seed3_YSnoiseZ_preds_ex1.npy")
# %%
new_preds = np.load("YSnoiseZ_4s_1ex_spdl_preds.npy")
comb_refs = np.concatenate([ref_preds1, ref_preds2, ref_preds3], axis=1)

# %%
plt.hist(comb_refs[0, :], bins=np.arange(-0.75, 0.80, 0.05))

# %%
plt.hist(new_preds[0, :], bins=np.arange(-0.75, 0.80, 0.05))
# %%
print(np.std(comb_refs))
print(np.std(new_preds))
# %%
print(np.sum(new_preds[0, :] == 0.75))
print(np.sum(new_preds[0, :] == -0.75))
# %%
