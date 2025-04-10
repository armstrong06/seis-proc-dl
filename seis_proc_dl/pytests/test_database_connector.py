import pytest
from sqlalchemy.orm import sessionmaker
from sqlalchemy import engine
from sqlalchemy import inspect
from datetime import datetime, timedelta
from copy import deepcopy
import numpy as np
from seis_proc_dl.apply_to_continuous.database_connector import DetectorDBConnection
from seis_proc_dl.apply_to_continuous.apply_detectors import ApplyDetector
from seis_proc_db.database import engine
from seis_proc_db import services, tables


datetimeformat = "%Y-%m-%dT%H:%M:%S.%f"
dateformat = "%Y-%m-%d"

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


@pytest.fixture
def contdatainfo_ex():
    new_date = datetime.strptime("2013-03-31", dateformat)
    metadata_dict = {
        "sampling_rate": 100.0,
        "dt": 0.01,
        "original_npts": 8280000,
        "original_starttime": datetime.strptime(
            "2013-03-31T01:00:00.00", datetimeformat
        ),
        "original_endtime": datetime.strptime("2013-03-31T23:59:59.59", datetimeformat),
        "npts": 8640000,
        "starttime": datetime.strptime("2013-03-31T00:00:00.00", datetimeformat),
        "previous_appended": False,
    }
    return new_date, deepcopy(metadata_dict)


@pytest.fixture
def simple_gaps_ex():
    gap1 = [
        # "Net",
        # "Stat",
        # "",
        "HHE",
        datetime.strptime("2013-03-31T01:00:00.00", datetimeformat),
        datetime.strptime("2013-03-31T02:00:00.00", datetimeformat),
        # 777,
        # 777,
    ]

    gap2 = [
        # "Net",
        # "Stat",
        # "",
        "HHE",
        datetime.strptime("2013-03-31T03:00:00.00", datetimeformat),
        datetime.strptime("2013-03-31T04:00:00.00", datetimeformat),
        # 777,
        # 777,
    ]

    gap3 = [
        # "Net",
        # "Stat",
        # "",
        "HHE",
        datetime.strptime("2013-03-31T05:00:00.00", datetimeformat),
        datetime.strptime("2013-03-31T06:00:00.00", datetimeformat),
        # 777,
        # 777,
    ]

    return deepcopy([gap1, gap2, gap3])


@pytest.fixture
def close_gaps_ex():
    gap1 = [
        # "Net",
        # "Stat",
        # "",
        "HHZ",
        datetime.strptime("2013-03-31T01:00:00.00", datetimeformat),
        datetime.strptime("2013-03-31T02:00:00.00", datetimeformat),
        # 777,
        # 777,
    ]

    # 1 second between gap1 and gap2
    gap2 = [
        # "Net",
        # "Stat",
        # "",
        "HHZ",
        datetime.strptime("2013-03-31T02:00:01.00", datetimeformat),
        datetime.strptime("2013-03-31T02:00:11.00", datetimeformat),
        # 777,
        # 777,
    ]

    # 1 second between gap2 and gap3
    gap3 = [
        # "Net",
        # "Stat",
        # "",
        "HHZ",
        datetime.strptime("2013-03-31T02:00:12.00", datetimeformat),
        datetime.strptime("2013-03-31T02:00:22.00", datetimeformat),
        # 777,
        # 777,
    ]

    # 5.0 seconds between gap3 and gap4
    gap4 = [
        # "Net",
        # "Stat",
        # "",
        "HHZ",
        datetime.strptime("2013-03-31T02:00:27.00", datetimeformat),
        datetime.strptime("2013-03-31T02:00:37.00", datetimeformat),
        # 777,
        # 777,
    ]

    return deepcopy([gap1, gap2, gap3, gap4])


@pytest.fixture
def detections_ex():
    d1 = {"sample": 1000, "height": 90, "width": 20}
    d2 = {"sample": 20000, "height": 70, "width": 30}
    d3 = {"sample": 30000, "height": 80, "width": 25}

    return deepcopy([d1, d2, d3])


class TestDetectorDBConnection:
    @pytest.fixture
    def db_session_with_3c_stat_loaded(self, db_session):
        session, patch_session = db_session  # Unpack session & patch function
        db_conn = DetectorDBConnection(3)  # Create instance

        patch_session(db_conn)  # Patch `self.Session` on the instance
        start, end = db_conn.get_channel_dates(
            datetime.strptime("2012-10-10T00:00:00.00", datetimeformat), "YNR", "HH"
        )

        return session, db_conn, start, end

    def test_get_channels(self, db_session_with_3c_stat_loaded):
        session, db_conn, start, end = db_session_with_3c_stat_loaded

        assert start == datetime.strptime(
            "2003-09-09T00:00:00.00", datetimeformat
        ), "invalid start"
        assert end == None, "invalid end"
        assert (
            len(db_conn.channel_info.channel_ids.keys()) == 3
        ), "invalid number of channels"
        assert db_conn.channel_info.ondate == datetime.strptime(
            "2011-09-11T00:00:00.00", datetimeformat
        ), "invalid channel ondate"
        assert db_conn.channel_info.offdate == datetime.strptime(
            "2013-03-31T23:59:59.00", datetimeformat
        ), "invalid channel offdate"

    def test_get_channels_1C(self, db_session):
        session, patch_session = db_session  # Unpack session & patch function
        db_conn = DetectorDBConnection(1)  # Create instance
        patch_session(db_conn)  # Patch `self.Session` on the instance

        start, end = db_conn.get_channel_dates(
            datetime.strptime("2012-10-10T00:00:00.00", datetimeformat), "QLMT", "EHZ"
        )

        assert start == datetime.strptime(
            "2001-06-09T00:00:00.00", datetimeformat
        ), "invalid start"
        assert end == None, "invalid end"
        assert (
            len(db_conn.channel_info.channel_ids.keys()) == 1
        ), "invalid number of channels"
        assert db_conn.channel_info.ondate == datetime.strptime(
            "2003-06-10T18:00:00.00", datetimeformat
        ), "invalid channel ondate"
        assert db_conn.channel_info.offdate == datetime.strptime(
            "2013-09-06T18:00:00.00", datetimeformat
        ), "invalid channel offdate"

    @pytest.fixture
    def db_session_with_detection_methods(self, db_session_with_3c_stat_loaded):
        # Unpack session & patch function
        session, db_conn, _, _ = db_session_with_3c_stat_loaded

        db_conn.add_detection_method("TEST-P", "test method", "data/path/P", "P")
        db_conn.add_detection_method("TEST-S", "test method", "data/path/S", "S")

        return session, db_conn

    def test_add_detection_method_P(self, db_session_with_detection_methods):
        session, db_conn = db_session_with_detection_methods

        assert (
            db_conn.p_detection_method_id is not None
        ), "detection_method id is not set"
        det_method = session.get(tables.DetectionMethod, db_conn.p_detection_method_id)
        assert det_method is not None, "No detection method returned"
        assert det_method.name == "TEST-P", "invalid name"
        assert det_method.phase == "P", "invalid phase"

    def test_add_detection_method_S(self, db_session_with_detection_methods):
        session, db_conn = db_session_with_detection_methods

        assert (
            db_conn.s_detection_method_id is not None
        ), "detection_method id is not set"
        det_method = session.get(tables.DetectionMethod, db_conn.s_detection_method_id)
        assert det_method is not None, "No detection method returned"
        assert det_method.name == "TEST-S", "invalid name"
        assert det_method.phase == "S", "invalid phase"

    def test_validate_channels_for_date_valid(self, db_session_with_3c_stat_loaded):
        session, db_conn, start, end = db_session_with_3c_stat_loaded
        date = datetime.strptime("2013-03-31", dateformat)
        valid = db_conn.validate_channels_for_date(date)
        assert valid, "Returned false"

    def test_validate_channels_for_date_invalid(self, db_session_with_3c_stat_loaded):
        session, db_conn, start, end = db_session_with_3c_stat_loaded
        date = datetime.strptime("2013-04-01", dateformat)
        valid = db_conn.validate_channels_for_date(date)
        assert not valid, "Returned true"

    def test_update_channels(self, db_session_with_3c_stat_loaded):
        session, db_conn, start, end = db_session_with_3c_stat_loaded
        date = datetime.strptime("2013-04-01", dateformat)
        db_conn.update_channels(date)
        assert db_conn.channel_info.ondate == datetime.strptime(
            "2013-04-01T00:00:00.00", datetimeformat
        ), "invalid ondate"
        assert db_conn.channel_info.offdate == datetime.strptime(
            "2015-08-25T23:59:59.00", datetimeformat
        ), "invalid ondate"

    def test_start_new_day_valid_date(self, db_session_with_3c_stat_loaded):
        session, db_conn, start, end = db_session_with_3c_stat_loaded

        new_date = datetime.strptime("2013-03-31", dateformat)
        db_conn.start_new_day(new_date)
        assert (
            db_conn.daily_info.date == new_date
        ), "invalid date in DailyDetectionDBInfo"
        assert db_conn.channel_info.ondate == datetime.strptime(
            "2011-09-11T00:00:00.00", datetimeformat
        ), "invalid channel ondate"
        assert db_conn.channel_info.offdate == datetime.strptime(
            "2013-03-31T23:59:59.00", datetimeformat
        ), "invalid channel offdate"

    def test_start_new_day_invalid_date(self, db_session_with_3c_stat_loaded):
        session, db_conn, start, end = db_session_with_3c_stat_loaded

        new_date = datetime.strptime("2013-04-01", dateformat)
        db_conn.start_new_day(new_date)
        assert (
            db_conn.daily_info.date == new_date
        ), "invalid date in DailyDetectionDBInfo"
        assert db_conn.channel_info.ondate == datetime.strptime(
            "2013-04-01T00:00:00.00", datetimeformat
        ), "invalid ondate"
        assert db_conn.channel_info.offdate == datetime.strptime(
            "2015-08-25T23:59:59.00", datetimeformat
        ), "invalid ondate"

    @pytest.fixture
    def db_session_with_saved_contdatainfo(
        self, db_session_with_detection_methods, contdatainfo_ex
    ):
        session, db_conn = db_session_with_detection_methods
        new_date, metadata_dict = contdatainfo_ex
        db_conn.save_data_info(new_date, metadata_dict)
        return session, db_conn

    def test_save_data_info(self, db_session_with_saved_contdatainfo, contdatainfo_ex):
        session, db_conn = db_session_with_saved_contdatainfo
        new_date, metadata_dict = contdatainfo_ex
        assert (
            db_conn.daily_info.date == new_date
        ), "invalid date in DailyDetectionDBInfo"
        assert db_conn.daily_info.contdatainfo_id is not None, "contdatainfo id not set"
        contdatainfo = session.get(
            tables.DailyContDataInfo, db_conn.daily_info.contdatainfo_id
        )
        assert inspect(contdatainfo).persistent, "contdatainfo not persistent"
        assert contdatainfo is not None, "contdatainfo not set"
        assert contdatainfo.chan_pref == "HH", "invalid chan_pref"
        assert contdatainfo.date == new_date.date(), "contdatainfo date incorrect"
        assert contdatainfo.proc_start == datetime.strptime(
            "2013-03-31T00:00:00.00", datetimeformat
        ), "invalid proc_start"

    def test_save_data_info_error(self, db_session_with_3c_stat_loaded):
        session, db_conn, start, end = db_session_with_3c_stat_loaded
        new_date = datetime.strptime("2013-03-31", dateformat)
        metadata_dict = None
        error = "no_data"
        db_conn.save_data_info(new_date, metadata_dict, error=error)
        assert (
            db_conn.daily_info.date == new_date
        ), "invalid date in DailyDetectionDBInfo"
        assert db_conn.daily_info.contdatainfo_id is not None, "contdatainfo id not set"
        contdatainfo = session.get(
            tables.DailyContDataInfo, db_conn.daily_info.contdatainfo_id
        )
        assert inspect(contdatainfo).persistent, "contdatainfo not persistent"
        assert contdatainfo is not None, "contdatainfo not set"
        assert contdatainfo.chan_pref == "HH", "invalid chan_pref"
        assert contdatainfo.date == new_date.date(), "contdatainfo date incorrect"
        assert contdatainfo.proc_start == None, "proc_start is not None"
        assert contdatainfo.error == error, "error incorrect"

    def test_save_data_info_duplicate_identical_entry(
        self, db_session_with_saved_contdatainfo, contdatainfo_ex
    ):
        session, db_conn = db_session_with_saved_contdatainfo
        new_date, metadata_dict = contdatainfo_ex

        # This should not throw any error because the duplicate rows are the same
        db_conn.save_data_info(new_date, metadata_dict)

    def test_save_data_info_duplicate_nonidentical_entry(
        self, db_session_with_saved_contdatainfo, contdatainfo_ex
    ):
        session, db_conn = db_session_with_saved_contdatainfo
        new_date, metadata_dict = contdatainfo_ex

        # Change a value in the metadata
        metadata_dict["npts"] = 100

        # This should throw any error because the duplicate rows are the different
        with pytest.raises(ValueError):
            db_conn.save_data_info(new_date, metadata_dict)

    @pytest.fixture
    def multi_channel_gaps_ex(self, close_gaps_ex, simple_gaps_ex):
        gaps = close_gaps_ex
        gaps += simple_gaps_ex

        return gaps

    def test_convert_gap_to_dict(self, simple_gaps_ex):
        gaps = simple_gaps_ex
        formatted = DetectorDBConnection.convert_gap_to_dict(gaps[0], 1, 2)

        assert formatted["data_id"] == 1
        assert formatted["chan_id"] == 2
        assert formatted["start"] == datetime.strptime(
            "2013-03-31T01:00:00.00", datetimeformat
        )
        assert formatted["end"] == datetime.strptime(
            "2013-03-31T02:00:00.00", datetimeformat
        )
        assert formatted["avail_sig_sec"] == 0.0

    def test_format_channel_gaps_simple(
        self, db_session_with_saved_contdatainfo, simple_gaps_ex
    ):
        session, db_conn = db_session_with_saved_contdatainfo
        gaps = simple_gaps_ex

        formatted = db_conn.format_channel_gaps(gaps, 1, 5)

        assert len(formatted) == 3, "incorrect number of gaps"

    def test_format_channel_gaps_merged(
        self, db_session_with_saved_contdatainfo, close_gaps_ex
    ):
        session, db_conn = db_session_with_saved_contdatainfo
        gaps = close_gaps_ex

        formatted = db_conn.format_channel_gaps(gaps, 1, 5)

        assert len(formatted) == 2, "incorrect number of gaps"
        assert (
            formatted[0]["avail_sig_sec"] == 2.0
        ), "incorrect avail_sig_sec for merged gap"
        assert (
            formatted[1]["avail_sig_sec"] == 0.0
        ), "incorrect avail_sig_sec for single gap"
        assert formatted[0]["start"] == datetime.strptime(
            "2013-03-31T01:00:00.00", datetimeformat
        ), "incorrect start for merged gap"
        assert formatted[0]["end"] == datetime.strptime(
            "2013-03-31T02:00:22.00", datetimeformat
        ), "incorrect end for merged gap"

    def test_format_and_save_gaps(
        self, db_session_with_saved_contdatainfo, multi_channel_gaps_ex
    ):
        session, db_conn = db_session_with_saved_contdatainfo
        gaps = multi_channel_gaps_ex

        db_conn.format_and_save_gaps(gaps, 5)
        gaps_E = services.get_gaps(
            session,
            db_conn.channel_info.channel_ids["HHE"],
            db_conn.daily_info.contdatainfo_id,
        )
        assert len(gaps_E) == 3, "Incorrect number of gaps on HHE channel"

        gaps_Z = services.get_gaps(
            session,
            db_conn.channel_info.channel_ids["HHZ"],
            db_conn.daily_info.contdatainfo_id,
        )
        assert len(gaps_Z) == 2, "Incorrect number of gaps on HHZ channel"

        # The only persistent objects in the session are gaps
        # from sqlalchemy import inspect
        # for obj in session:
        #     print(obj)
        # ContDataInfo is detached
        # print("persistent", inspect(db_conn.daily_info.contdatainfo).persistent)
        # print("detached", inspect(db_conn.daily_info.contdatainfo).detached)

    def test_get_dldet_fk_ids_P(self, db_session_with_saved_contdatainfo):
        session, db_conn = db_session_with_saved_contdatainfo

        d = db_conn.get_dldet_fk_ids(is_p=True)
        contdata = session.get(tables.DailyContDataInfo, d["data"])
        assert contdata is not None, "contdatainfo not found"
        assert contdata.id is not None, "contatainfo.id is not set"
        p_det_meth = session.get(tables.DetectionMethod, d["method"])
        assert p_det_meth is not None, "p detection_method is not set"
        assert p_det_meth.phase == "P", "p detection_method phase is invalid"

    def test_get_dldet_fk_ids_S(self, db_session_with_saved_contdatainfo):
        session, db_conn = db_session_with_saved_contdatainfo

        d = db_conn.get_dldet_fk_ids(is_p=False)
        contdata = session.get(tables.DailyContDataInfo, d["data"])
        assert contdata is not None, "contdatainfo not found"
        assert contdata.id is not None, "contatainfo.id is not set"
        s_det_meth = session.get(tables.DetectionMethod, d["method"])
        assert s_det_meth is not None, "S detection_method is not set"
        assert s_det_meth.phase == "S", "S detection_method phase is invalid"

    @pytest.fixture
    def db_session_with_P_dldets(
        self, db_session_with_saved_contdatainfo, detections_ex
    ):
        session, db_conn = db_session_with_saved_contdatainfo
        det_list = detections_ex
        ids = db_conn.get_dldet_fk_ids(is_p=True)
        for det in det_list:
            det["phase"] = "P"
            det["data_id"] = ids["data"]
            det["method_id"] = ids["method"]

        db_conn.save_detections(det_list)

        return session, db_conn

    def test_save_detections(self, db_session_with_P_dldets):
        session, db_conn = db_session_with_P_dldets
        ids = db_conn.get_dldet_fk_ids(is_p=True)
        selected_dets = services.get_dldetections(
            session, ids["data"], ids["method"], 0.0, phase="P"
        )

        assert len(selected_dets) == 3, "incorrect number of detections"

    def test_save_picks_from_detections(self, db_session_with_P_dldets):
        session, db_conn = db_session_with_P_dldets

        pick_thresh = 75
        auth = "TEST"
        wf_filt_low = None
        wf_filt_high = None
        wf_proc_notes = "TEST DATA"
        seconds_around_pick = 10

        cont_data = np.zeros((50000, 3))
        samples = int(seconds_around_pick * 100)
        cont_data[1000 - samples : 1000 + samples + 1] = 1
        cont_data[20000 - samples : 20000 + samples + 1] = 2
        cont_data[30000 - samples : 30000 + samples + 1] = 3

        db_conn.save_picks_from_detections(
            pick_thresh=pick_thresh,
            is_p=True,
            auth=auth,
            continuous_data=cont_data,
            wf_filt_low=None,
            wf_filt_high=None,
            wf_proc_notes=wf_proc_notes,
            seconds_around_pick=seconds_around_pick,
        )

        picks = services.get_picks(session, db_conn.station_id, "HH", phase="P")
        assert len(picks) == 2, "incorrect number of picks"

        for pick in picks:
            det = session.get(tables.DLDetection, pick.detid)
            assert det.height > pick_thresh
            contdatainfo = session.get(tables.DailyContDataInfo, det.data_id)

            assert (
                pick.ptime - timedelta(seconds=(det.sample / contdatainfo.samp_rate))
                == contdatainfo.proc_start
            ), "invalid pick time"

            assert pick.auth == "TEST", "invalid author"
            assert pick.phase == "P"
            assert pick.chan_pref == "HH"
            assert pick.sta_id == db_conn.station_id
            assert pick.snr is None
            assert pick.amp is None

            wf = services.get_waveforms(session, pick.id)
            assert len(wf) == 3, "invalid wf size"

            assert det.sample == 1000 or det.sample == 30000, "incorrect dets saved"
            if det.sample == 1000:
                assert np.all(
                    np.array(wf[0].data) == 1
                ), "invalid data for wf[0] when det.sample == 1000"
                assert np.all(
                    np.array(wf[1].data) == 1
                ), "invalid data for wf[1] when det.sample == 1000"
                assert np.all(
                    np.array(wf[2].data) == 1
                ), "invalid data for wf[2] when det.sample == 1000"
            elif det.sample == 30000:
                assert np.all(
                    np.array(wf[0].data) == 3
                ), "invalid data for wf[0] when det.sample == 30000"
                assert np.all(
                    np.array(wf[1].data) == 3
                ), "invalid data for wf[1] when det.sample == 30000"
                assert np.all(
                    np.array(wf[2].data) == 3
                ), "invalid data for wf[2] when det.sample == 30000"

            assert len(wf[0].data) == samples * 2 + 1, "invalid data length for wf[0]"
            assert len(wf[1].data) == samples * 2 + 1, "invalid data length for wf[1]"
            assert len(wf[2].data) == samples * 2 + 1, "invalid data length for wf[2]"
            assert wf[0].start == pick.ptime - timedelta(
                seconds=seconds_around_pick
            ), "invalid start for wf[0]"
            assert wf[1].start == pick.ptime - timedelta(
                seconds=seconds_around_pick
            ), "invalid start for wf[1]"
            assert wf[2].start == pick.ptime - timedelta(
                seconds=seconds_around_pick
            ), "invalid start for wf[2]"
            assert wf[0].end == pick.ptime + timedelta(
                seconds=seconds_around_pick + 0.01
            ), "invalid end for wf[0]"
            assert wf[1].end == pick.ptime + timedelta(
                seconds=seconds_around_pick + 0.01
            ), "invalid end for wf[1]"
            assert wf[2].end == pick.ptime + timedelta(
                seconds=seconds_around_pick + 0.01
            ), "invalid end for wf[2]"
            assert wf[0].filt_low is None
            assert wf[1].filt_low is None
            assert wf[2].filt_low is None
            assert wf[0].filt_high is None
            assert wf[1].filt_high is None
            assert wf[2].filt_high is None


examples_dir = "/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/seis_proc_dl/pytests/example_files"
models_path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"
apply_detectors_outdir = f"{examples_dir}/applydetector_results"

apply_detector_config = {
    "paths": {
        "data_dir": examples_dir,
        "output_dir": apply_detectors_outdir,
        "one_comp_p_model": f"{models_path}/oneCompPDetectorMEW_model_022.pt",
        "three_comp_p_model": f"{models_path}/pDetectorMew_model_026.pt",
        "three_comp_s_model": f"{models_path}/sDetector_model032.pt",
    },
    "unet": {
        "window_length": 1008,
        "sliding_interval": 500,
        "device": "cpu",
        "min_torch_threads": 2,
        "min_presigmoid_value": -70,
        "batchsize": 256,
        "use_openvino": False,
        "post_probs_file_type": "MSEED",
    },
    "dataloader": {
        "store_N_seconds": 10,
        # "expected_file_duration_s":3600,
        "min_signal_percent": 0,
    },
    "database": {
        "det_method_1c_P": {"name": "TEST_1C_UNET", "desc": "test for 1C P dets"},
        "det_method_3c_P": {"name": "TEST_3C_UNET", "desc": "test for 3C P dets"},
        "det_method_3c_S": {"name": "TEST_3C_UNET", "desc": "test for 3C S dets"},
        "p_det_thresh": 50,
        "s_det_thresh": 50,
        "p_pick_thresh": 75,
        "s_pick_thresh": 75,
        "wf_seconds_around_pick": 10,
        "pick_author": "SPDL",
        "min_gap_separation_seconds": 5,
    },
}


class TestApplyDetectorDB:
    def test_init_3c(self, db_session):
        session, _ = db_session
        applier = ApplyDetector(
            3, apply_detector_config, session_factory=lambda: session
        )
        assert applier.db_conn is not None, "db_conn not defined"
        assert applier.db_conn.ncomps == 3, "db_conn.ncomps incorrect"
        assert (
            applier.db_conn.p_detection_method_id is not None
        ), "p detection method not set"
        assert (
            applier.db_conn.s_detection_method_id is not None
        ), "s detection method not set"
        assert applier.p_det_thresh == 50
        assert applier.s_det_thresh == 50
        assert applier.p_pick_thresh == 75
        assert applier.s_pick_thresh == 75
        assert applier.wf_seconds_around_pick == 10.0
        assert applier.db_pick_author == "SPDL"

    def test_init_1c(self, db_session):
        session, _ = db_session
        applier = ApplyDetector(
            1, apply_detector_config, session_factory=lambda: session
        )
        assert applier.db_conn is not None, "db_conn not defined"
        assert applier.db_conn.ncomps == 1, "db_conn.ncomps incorrect"
        assert (
            applier.db_conn.p_detection_method_id is not None
        ), "p detection method not set"
        assert (
            applier.db_conn.s_detection_method_id is None
        ), "s detection method should not set"
        assert applier.p_det_thresh == 50
        assert applier.s_det_thresh is None
        assert applier.p_pick_thresh == 75
        assert applier.s_pick_thresh is None
        assert applier.wf_seconds_around_pick == 10.0
        assert applier.db_pick_author == "SPDL"
