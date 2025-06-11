import numpy as np
import pytest
from unittest import mock
import shutil
from datetime import datetime, timedelta
import h5py
import os
from sqlalchemy.orm import sessionmaker
from seis_proc_dl.apply_to_continuous import apply_swag_pickers
from seis_proc_db import services, pytables_backend
from seis_proc_db.database import engine

datetimeformat = "%Y-%m-%dT%H:%M:%S.%f"


@pytest.fixture
def mock_pytables_config():
    print(os.getcwd())
    d = "/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/seis_proc_dl/pytests/repicker_slurm/pytables_outputs"
    shutil.rmtree(d)
    with mock.patch(
        "seis_proc_db.pytables_backend.HDF_BASE_PATH",
        d,
    ):
        yield


# Create a session factory (not bound yet)
TestSessionFactory = sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def db_session(monkeypatch):
    """
    Provides a test database session that rolls back all changes.
    I got this from chatgpt because I am confused about this...
    """

    # Connect to the database
    connection = engine.connect()

    # Begin a non-ORM transaction (acts like a savepoint)
    trans = connection.begin()

    # Define a callable session factory with rollback protection
    def session_factory():
        return TestSessionFactory(
            bind=connection, join_transaction_mode="create_savepoint"
        )

    # Create a function to patch `self.Session` for instances of `DetectorDBConnection`
    def patch_session(instance):
        monkeypatch.setattr(instance, "Session", lambda: session_factory())

    session = session_factory()  # Create an initial session
    yield session, patch_session  # Provide session & patch function to the test

    # Teardown: Rollback all changes & close the connection
    session.close()
    trans.rollback()
    connection.close()


examples_dir = "/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/seis_proc_dl/pytests/example_files/repicker_slurm"


@pytest.fixture
def db_session_with_waveform_info(db_session, mock_pytables_config):
    db_session, _ = db_session
    ids = {}
    # Insert the stations
    sta_dict = {
        "ondate": datetime.strptime("2010-01-01T00:00:00.00", datetimeformat),
        "lat": 44.7155,
        "lon": -110.67917,
        "elev": 2336,
    }
    sta1 = services.insert_station(db_session, "JK", "TST1", **sta_dict)
    db_session.flush()

    # Insert the channels
    chan_info = {
        "loc": "01",
        "ondate": datetime.strptime("2010-01-01T00:00:00.00", datetimeformat),
        "samp_rate": 100.0,
        "clock_drift": 1e-5,
        "sensor_desc": "Nanometrics something or other",
        "sensit_units": "M/S",
        "sensit_val": 9e9,
        "sensit_freq": 5,
        "lat": 44.7155,
        "lon": -110.67917,
        "elev": 2336,
        "depth": 100,
        "azimuth": 90,
        "dip": -90,
        "offdate": None,
        "overall_gain_vel": None,
    }
    all_channel_dict = {}
    for id in [sta1.id]:
        for code in ["HHZ"]:
            chan_info["seed_code"] = code
            chan_info["sta_id"] = id
            all_channel_dict[f"{id}.{code}"] = services.insert_channel(
                db_session, chan_info
            )

    db_session.flush()

    # Insert P Picks
    p_dict = {
        "chan_pref": "HH",
        "phase": "P",
        "ptime": datetime.strptime("2010-02-01T00:02:00.00", datetimeformat),
        "auth": "TEST",
    }
    p1 = services.insert_pick(db_session, sta1.id, **p_dict)

    # Insert waveform sources
    wf_source1 = services.insert_waveform_source(
        db_session, "TEST-ExtractContData", "Extract snippets"
    )

    db_session.flush()

    ids["p_pick1"] = p1.id
    ids["wf_source1"] = wf_source1.id

    try:
        # Open waveform storages
        wf_storages = {}
        for code in ["HHZ", "HHE", "HHN"]:
            for phase in ["P", "S"]:
                wf_storage = pytables_backend.WaveformStorage(
                    expected_array_length=400,
                    net="JK",
                    sta=str(id),
                    loc="01",
                    seed_code=code,
                    ncomps=3,
                    phase=phase,
                    wf_source_id=wf_source1.id,
                )
                wf_storages[f"{id}.{code}.{phase}.{wf_source1.id}"] = wf_storage

        ### Insert waveform infos ###

        def insert_wf_info(
            phase, pick, chan_code, wf_source, data, start_ind=None, end_ind=None
        ):
            _ = services.insert_waveform_pytable(
                db_session,
                wf_storages[f"{pick.sta_id}.{chan_code}.{phase}.{wf_source.id}"],
                all_channel_dict[f"{pick.sta_id}.{chan_code}"].id,
                pick.id,
                wf_source.id,
                start=pick.ptime - timedelta(seconds=2),
                end=pick.ptime + timedelta(seconds=2),
                data=data,
                signal_start_ind=start_ind,
                signal_end_ind=end_ind,
            )

        # TODO: I should load the examples from Ben...

        with h5py.File("{examples_dir}YSnoiseZ_4s_1ex.h5") as f:
            p_data = f["X"][0, :]

        assert p_data.shape == (400,)
        insert_wf_info("P", p1, "HHZ", wf_source1, p_data)

        db_session.commit()
    finally:
        for _, wf_storage in wf_storages.items():
            wf_storage.commit()
            if wf_storage._is_open:
                wf_storage.close()
    return db_session, ids


@pytest.fixture
def method_dicts_ex():
    repicker_dict = {
        "name": "TEST-MSWAG-BSSA-2023",
        "desc": "MSWAG models presented in Armstrong et al., 2023.",
    }
    cal_dict = {
        "name": "TEST-Kuleshov",
        "desc": (
            "Empirical calibration using Kuleshov et al., 2018 approach. Calibration was used in Armstrong et al. 2023."
            "Uses the trimmed (inner fence) median and std dev of the combined 3 swag model predictions."
        ),
    }

    return repicker_dict, cal_dict


def test_apply_noise(db_session_with_waveform_info, method_dicts_ex):
    examples_dir = "/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/seis_proc_dl/pytests/example_files"
    is_p = True
    device = "cuda:0"
    train_path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/swag_info"
    train_file = "p_uuss_train_4s_1dup.h5"
    train_bs = 1024
    train_n_workers = 4
    shuffle_train = False
    data_file = "YSnoiseZ_4s_1ex.h5"
    outfile = "./YSnoiseZ_4s_1ex_spdl_preds"
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
    dur = 400  # samples
    start_date = datetime.strptime("2010-02-01T00:00:00.00", datetimeformat)
    end_date = datetime.strptime("2010-02-02T00:00:00.00", datetimeformat)

    # Initialize the picker
    sp = apply_swag_pickers.MultiSWAGPickerDB(
        is_p_picker=is_p,
        swag_model_dir=model_path,
        cal_model_file=".",
        device=device,
    )
    repicker_dict, cal_dict = method_dicts_ex
    sp.start_db_conn(
        repicker_dict=repicker_dict,
        cal_dict=cal_dict,
    )
    # Load the training data for bn_updates
    train_loader = sp.torch_loader(
        train_file,
        train_path,
        train_bs,
        train_n_workers,
        shuffle=shuffle_train,
    )
    # Load the new estimated picks
    ids, data_loader = sp.torch_loader_from_db(
        n_samples=dur,
        batch_size=data_bs,
        num_workers=data_n_workers,
        start_date=start_date,
        end_date=end_date,
        wf_source_list=["TEST-ExtractContData"],
        padding=0,
        no_proc=True,
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
    # new_preds = sp.apply_picker(ensemble, data_loader, train_loader, N)

    # np.save(outfile, new_preds)

    # assert np.std(new_preds > 0.3), "std for noise is smaller than expected"
    # assert np.sum(abs(new_preds[0, :]) == 0.75) > 0, "expected some predictions at /pm 0.75"
