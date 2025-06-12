import numpy as np
from seis_proc_dl.apply_to_continuous import apply_swag_pickers

examples_dir = "/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/seis_proc_dl/pytests/example_files"
is_p = True
device = "cuda:0"
train_path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/swag_info"
train_file = "p_uuss_train_4s_1dup.h5"
train_bs = 1024
train_n_workers = 4
shuffle_train = False
data_file = "YSnoiseZ_4s_1ex_norm.h5"
outfile = "./YSnoiseZ_4s_1ex_norm_spdl_preds"
# data_file = "uuss_NGB_4s_1ex.h5"
# outfile = "./uuss_NGB_4s_1ex_spdl_preds"
data_path = f"{examples_dir}/repicker_test_exs"
data_bs = 1
data_n_workers = 1
n_data_examples = -1
model_path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"
swag_model1 = "pPicker_swag60_seed1.pt"
swag_model2 = "pPicker_swag60_seed2.pt"
swag_model3 = "pPicker_swag60_seed3.pt"
seeds = [1, 2, 3]
cov_mat = True
K = 20
N = 40

# Initialize the picker
sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=is_p, device=device)
# Load the training data for bn_updates
train_loader = sp.torch_loader(
    train_file,
    train_path,
    train_bs,
    train_n_workers,
    shuffle=shuffle_train,
)
# Load the new estimated picks
data_loader = sp.torch_loader(
    data_file,
    data_path,
    data_bs,
    data_n_workers,
    shuffle=False,
    n_examples=n_data_examples,
)
# Load the MultiSWAG ensemble
ensemble = sp.load_swag_ensemble(
    swag_model1,
    swag_model2,
    swag_model3,
    seeds,
    cov_mat,
    K,
    swag_model_dir=model_path,
)
new_preds = sp.apply_picker(ensemble, data_loader, train_loader, N)

np.save(outfile, new_preds)

# ref_preds1 = np.load("../example_files/repicker_test_exs/seed1_NGB_preds_ex1.npy")
# ref_preds2 = np.load("../example_files/repicker_test_exs/seed2_NGB_preds_ex1.npy")
# ref_preds3 = np.load("../example_files/repicker_test_exs/seed3_NGB_preds_ex1.npy")

# assert np.allclose(ref_preds1, new_preds[0:1, 0:40]), "seed1 not close"
# assert np.allclose(ref_preds2, new_preds[0:1, 40:80]), "seed2 not close"
# assert np.allclose(ref_preds3, new_preds[0:1, 80:]), "seed3 not close"
