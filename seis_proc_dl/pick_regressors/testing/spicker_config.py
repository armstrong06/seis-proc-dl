"""Model config in json format for testing 3C, P-picker """

CFG = {
    "data":{
        "time_series_length":600, 
        "dt":0.01,
        "n_duplicates":2,
        "max_dt":0.7
    },
    "train": {
        "torch_device": "cuda:0",
        "batch_size": 32, 
        "epochs": 2, 
        "learning_rate": 0.00001, 
        "train_hdf5_file": "/home/armstrong/Research/newer/sPicker/uuss_data/uuss_NGB.h5",
        "validation_hdf5_file": "/home/armstrong/Research/newer/sPicker/uuss_data/uuss_NGB.h5", 
        "model_out_directory": "/home/armstrong/Research/git_repos/seis-proc-dl/pick_regressors/testing"
    }, 
    "model":{
        "num_channels":3, 
        "phase_type":"S",
        "max_dt_nn":0.85, 
        "freeze_convolutional_layers":False,
        "random_seed":2412045,
    }
}