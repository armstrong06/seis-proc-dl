# %%
import numpy as np

# %%
ref_preds1 = np.load("../example_files/repicker_test_exs/seed1_NGB_preds_ex1.npy")
ref_preds2 = np.load("../example_files/repicker_test_exs/seed2_NGB_preds_ex1.npy")
ref_preds3 = np.load("../example_files/repicker_test_exs/seed3_NGB_preds_ex1.npy")
# %%
new_preds = np.load("uuss_NGB_4s_1ex_spdl_preds.npy")

# %%
assert np.allclose(ref_preds1, new_preds[0:1, 0:40], atol=5e-2), "seed1 not close"
assert np.allclose(ref_preds2, new_preds[0:1, 40:80], atol=5e-2), "seed2 not close"
assert np.allclose(ref_preds3, new_preds[0:1, 80:], atol=5e-2), "seed3 not close"
# %%

assert np.allclose(
    np.std(ref_preds1), np.std(new_preds[0:1, 0:40]), atol=1e-3
), "seed1 std not close"
assert np.allclose(
    np.mean(ref_preds1), np.mean(new_preds[0:1, 0:40]), atol=1e-3
), "seed1 mean not close"
# %%
assert np.allclose(
    np.std(ref_preds2), np.std(new_preds[0:1, 40:80]), atol=1e-3
), "seed2 std not close"
assert np.allclose(
    np.mean(ref_preds2), np.mean(new_preds[0:1, 40:80]), atol=1e-3
), "seed2 mean not close"
# %%
assert np.allclose(
    np.std(ref_preds3), np.std(new_preds[0:1, 80:]), atol=1e-3
), "seed3 std not close"
assert np.allclose(
    np.mean(ref_preds3), np.mean(new_preds[0:1, 80:]), atol=1e-3
), "seed3 mean not close"
# %%
comb_refs = np.concatenate([ref_preds1, ref_preds2, ref_preds3], axis=1)

assert np.allclose(
    np.std(comb_refs), np.std(new_preds), atol=1e-3
), "combined std not close"
assert np.allclose(
    np.mean(comb_refs), np.mean(new_preds), atol=1e-3
), "combined mean not close"
# %%
