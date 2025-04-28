import pytest
from sqlalchemy.orm import sessionmaker
from sqlalchemy import engine
from sqlalchemy import inspect
from datetime import datetime, timedelta
from copy import deepcopy
import numpy as np
import os
from unittest import mock

from obspy.core import UTCDateTime as UTC
from seis_proc_dl.apply_to_continuous.database_connector import DetectorDBConnection
from seis_proc_dl.apply_to_continuous.apply_detectors import ApplyDetector
from seis_proc_db.database import engine
from seis_proc_db import services, tables


datetimeformat = "%Y-%m-%dT%H:%M:%S.%f"
dateformat = "%Y-%m-%d"

# Create a session factory (not bound yet)
TestSessionFactory = sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def mock_pytables_config():
    with mock.patch(
        "seis_proc_db.pytables_backend.HDF_BASE_PATH",
        "./pytests/pytables_outputs",
    ):
        yield


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
    new_date = datetime.strptime("2011-09-10", dateformat)
    metadata_dict = {
        "sampling_rate": 100.0,
        "dt": 0.01,
        "original_npts": 8280000,
        "original_starttime": datetime.strptime(
            "2011-09-10T01:00:00.00", datetimeformat
        ),
        "original_endtime": datetime.strptime("2011-09-10T23:59:59.59", datetimeformat),
        "npts": 8640000,
        "starttime": datetime.strptime("2011-09-10T00:00:00.00", datetimeformat),
        "previous_appended": False,
    }
    return new_date, deepcopy(metadata_dict)


@pytest.fixture
def simple_gaps_ex():
    gap1 = [
        "HHE",
        datetime.strptime("2011-09-10T01:00:00.00", datetimeformat),
        datetime.strptime("2011-09-10T02:00:00.00", datetimeformat),
    ]

    gap2 = [
        "HHE",
        datetime.strptime("2011-09-10T03:00:00.00", datetimeformat),
        datetime.strptime("2011-09-10T04:00:00.00", datetimeformat),
    ]

    gap3 = [
        "HHE",
        datetime.strptime("2011-09-10T05:00:00.00", datetimeformat),
        datetime.strptime("2011-09-10T06:00:00.00", datetimeformat),
    ]

    return deepcopy([gap1, gap2, gap3])


@pytest.fixture
def close_gaps_ex():
    gap1 = [
        # "Net",
        # "Stat",
        # "",
        "HHZ",
        datetime.strptime("2011-09-10T01:00:00.00", datetimeformat),
        datetime.strptime("2011-09-10T02:00:00.00", datetimeformat),
        # 777,
        # 777,
    ]

    # 1 second between gap1 and gap2
    gap2 = [
        "HHZ",
        datetime.strptime("2011-09-10T02:00:01.00", datetimeformat),
        datetime.strptime("2011-09-10T02:00:11.00", datetimeformat),
    ]

    # 1 second between gap2 and gap3
    gap3 = [
        "HHZ",
        datetime.strptime("2011-09-10T02:00:12.00", datetimeformat),
        datetime.strptime("2011-09-10T02:00:22.00", datetimeformat),
    ]

    # 5.0 seconds between gap3 and gap4
    gap4 = [
        "HHZ",
        datetime.strptime("2011-09-10T02:00:27.00", datetimeformat),
        datetime.strptime("2011-09-10T02:00:37.00", datetimeformat),
    ]

    return deepcopy([gap1, gap2, gap3, gap4])


@pytest.fixture
def detections_ex():
    d1 = {"sample": 1000, "height": 90, "width": 20, "inference_id": None}
    d2 = {"sample": 20000, "height": 70, "width": 30, "inference_id": None}
    d3 = {"sample": 30000, "height": 80, "width": 25, "inference_id": None}
    d4 = {"sample": 8639500, "height": 90, "width": 15, "inference_id": None}

    return deepcopy([d1, d2, d3, d4])


class TestDetectorDBConnection:
    @pytest.fixture
    def db_session_with_3c_stat_loaded(self, db_session):
        session, patch_session = db_session  # Unpack session & patch function
        db_conn = DetectorDBConnection(3)  # Create instance

        patch_session(db_conn)  # Patch `self.Session` on the instance
        start, end = db_conn.get_channel_dates(
            datetime.strptime("2011-09-10T00:00:00.00", datetimeformat),
            "WY",
            "YNR",
            "",
            "HH",
        )

        return session, db_conn, start, end

    def test_get_channels(self, db_session_with_3c_stat_loaded):
        session, db_conn, start, end = db_session_with_3c_stat_loaded

        assert start == datetime.strptime(
            "2003-09-09T00:00:00.00", datetimeformat
        ), "invalid start"
        assert end == datetime.strptime(
            "2013-03-31T23:59:59.00", datetimeformat
        ), "invalid end"
        assert (
            len(db_conn.channel_info.channel_ids.keys()) == 3
        ), "invalid number of channels"
        assert db_conn.channel_info.ondate == datetime.strptime(
            "2010-08-21T00:00:00.00", datetimeformat
        ), "invalid channel ondate"
        assert db_conn.channel_info.offdate == datetime.strptime(
            "2011-09-10T23:59:59.00", datetimeformat
        ), "invalid channel offdate"

    def test_get_channel_dates_YMV(self, db_session):
        session, patch_session = db_session  # Unpack session & patch function
        db_conn = DetectorDBConnection(3)  # Create instance

        patch_session(db_conn)  # Patch `self.Session` on the instance
        start, end = db_conn.get_channel_dates(
            datetime(2023, 1, 1), "WY", "YMV", "01", "HH"
        )

        assert start == datetime(2023, 8, 10, 0, 0, 0), "invalid start date"
        assert end is None, "invalid end date"

    def test_get_channel_dates_YJC(self, db_session):
        session, patch_session = db_session  # Unpack session & patch function
        db_conn = DetectorDBConnection(3)  # Create instance

        patch_session(db_conn)  # Patch `self.Session` on the instance
        start, end = db_conn.get_channel_dates(
            datetime(2023, 1, 1), "WY", "YJC", "01", "HH"
        )

        assert start == datetime(2023, 8, 8, 14, 0, 0), "invalid start date"
        assert end is None, "invalid end date"

    def test_get_channels_1C(self, db_session):
        session, patch_session = db_session  # Unpack session & patch function
        db_conn = DetectorDBConnection(1)  # Create instance
        patch_session(db_conn)  # Patch `self.Session` on the instance

        start, end = db_conn.get_channel_dates(
            datetime.strptime("2012-10-10T00:00:00.00", datetimeformat),
            "MB",
            "QLMT",
            "",
            "EHZ",
        )
        assert start == datetime.strptime(
            "2001-06-09T00:00:00.00", datetimeformat
        ), "invalid start"
        assert end == datetime.strptime(
            "2019-03-21T16:40:00.00", datetimeformat
        ), "invalid end"
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
        date = datetime.strptime("2011-09-10", dateformat)
        valid = db_conn.validate_channels_for_date(date)
        assert valid, "Returned false"

    def test_validate_channels_for_date_invalid(self, db_session_with_3c_stat_loaded):
        session, db_conn, start, end = db_session_with_3c_stat_loaded
        date = datetime.strptime("2011-09-11", dateformat)
        valid = db_conn.validate_channels_for_date(date)
        assert not valid, "Returned true"

    def test_update_channels_invalid(self, db_session_with_3c_stat_loaded):
        session, db_conn, start, end = db_session_with_3c_stat_loaded
        date = datetime.strptime("2013-04-01", dateformat)
        db_conn.update_channels(date)
        assert (
            db_conn.channel_info.ondate is None
        ), "ondate should not be set, date out of range"
        assert (
            db_conn.channel_info.offdate is None
        ), "offdate should not be set, date out of range"

    def test_update_channels(self, db_session_with_3c_stat_loaded):
        session, db_conn, start, end = db_session_with_3c_stat_loaded
        date = datetime.strptime("2011-09-11", dateformat)
        db_conn.update_channels(date)
        assert db_conn.channel_info.ondate == datetime.strptime(
            "2011-09-11T00:00:00.00", datetimeformat
        ), "invalid ondate"
        assert db_conn.channel_info.offdate == datetime.strptime(
            "2013-03-31T23:59:59.00", datetimeformat
        ), "invalid ondate"

    def test_start_new_day_valid_date(self, db_session_with_3c_stat_loaded):
        session, db_conn, start, end = db_session_with_3c_stat_loaded

        new_date = datetime.strptime("2011-09-10", dateformat)
        db_conn.start_new_day(new_date)
        assert (
            db_conn.daily_info.date == new_date
        ), "invalid date in DailyDetectionDBInfo"
        assert db_conn.channel_info.ondate == datetime.strptime(
            "2010-08-21T00:00:00.00", datetimeformat
        ), "invalid channel ondate"
        assert db_conn.channel_info.offdate == datetime.strptime(
            "2011-09-10T23:59:59.00", datetimeformat
        ), "invalid channel offdate"

    def test_start_new_day_invalid_date(self, db_session_with_3c_stat_loaded):
        session, db_conn, start, end = db_session_with_3c_stat_loaded

        new_date = datetime.strptime("2011-09-11", dateformat)
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

    def test_start_new_day_out_of_range(self, db_session_with_3c_stat_loaded):
        session, db_conn, start, end = db_session_with_3c_stat_loaded

        new_date = datetime.strptime("2013-04-01", dateformat)
        db_conn.start_new_day(new_date)
        assert (
            db_conn.daily_info.date == new_date
        ), "invalid date in DailyDetectionDBInfo"
        assert db_conn.channel_info.ondate is None, "invalid ondate"
        assert db_conn.channel_info.offdate is None, "invalid ondate"

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
            "2011-09-10T00:00:00.00", datetimeformat
        ), "invalid proc_start"

    def test_save_data_info_error(self, db_session_with_3c_stat_loaded):
        session, db_conn, start, end = db_session_with_3c_stat_loaded
        new_date = datetime.strptime("2011-09-10", dateformat)
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

    def test_save_data_info_insufficient_data(
        self, db_session_with_3c_stat_loaded, contdatainfo_ex
    ):
        session, db_conn, start, end = db_session_with_3c_stat_loaded
        new_date, metadata_dict = contdatainfo_ex
        error = "insufficient_data"
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
        assert contdatainfo.orig_start == datetime.strptime(
            "2011-09-10T01:00:00.00", datetimeformat
        ), "invalid orig_start"
        assert contdatainfo.proc_start is None, "invalid proc_start"
        assert contdatainfo.proc_npts is None, "invalid proc_npts"
        assert contdatainfo.proc_end is None, "invalid proc_end"
        assert contdatainfo.prev_appended is False, "invalud prev_appended"
        assert contdatainfo.error == error, "invalid error"

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

    def test_save_data_info_duplicate_nonidentical_entry_error(
        self, db_session_with_saved_contdatainfo, contdatainfo_ex
    ):
        session, db_conn = db_session_with_saved_contdatainfo
        new_date, _ = contdatainfo_ex

        # This should throw any error because the duplicate rows are the different
        with pytest.raises(ValueError):
            db_conn.save_data_info(new_date, None, error="no_data")

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
            "2011-09-10T01:00:00.00", datetimeformat
        )
        assert formatted["end"] == datetime.strptime(
            "2011-09-10T02:00:00.00", datetimeformat
        )
        assert formatted["avail_sig_sec"] == 0.0

    def test_format_channel_gaps_simple(
        self, db_session_with_saved_contdatainfo, simple_gaps_ex
    ):
        session, db_conn = db_session_with_saved_contdatainfo
        gaps = simple_gaps_ex

        formatted = db_conn.format_channel_gaps(gaps, 1, 5)

        assert len(formatted) == 3, "incorrect number of gaps"

    def test_format_channel_gaps_empty(self, db_session_with_saved_contdatainfo):
        session, db_conn = db_session_with_saved_contdatainfo

        formatted = db_conn.format_channel_gaps([], 1, 5)

        assert len(formatted) == 0, "incorrect number of gaps"

    def test_format_channel_gaps_none(self, db_session_with_saved_contdatainfo):
        session, db_conn = db_session_with_saved_contdatainfo

        formatted = db_conn.format_channel_gaps(None, 1, 5)

        assert len(formatted) == 0, "incorrect number of gaps"

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
            "2011-09-10T01:00:00.00", datetimeformat
        ), "incorrect start for merged gap"
        assert formatted[0]["end"] == datetime.strptime(
            "2011-09-10T02:00:22.00", datetimeformat
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

    def test_format_and_save_gaps_empty(
        self, db_session_with_saved_contdatainfo, multi_channel_gaps_ex
    ):
        session, db_conn = db_session_with_saved_contdatainfo
        gaps = []

        db_conn.format_and_save_gaps(gaps, 5)
        gaps_E = services.get_gaps(
            session,
            db_conn.channel_info.channel_ids["HHE"],
            db_conn.daily_info.contdatainfo_id,
        )
        assert len(gaps_E) == 0, "Incorrect number of gaps on HHE channel"

        gaps_Z = services.get_gaps(
            session,
            db_conn.channel_info.channel_ids["HHZ"],
            db_conn.daily_info.contdatainfo_id,
        )
        assert len(gaps_Z) == 0, "Incorrect number of gaps on HHZ channel"

    def test_format_and_save_gaps_none(
        self, db_session_with_saved_contdatainfo, multi_channel_gaps_ex
    ):
        session, db_conn = db_session_with_saved_contdatainfo
        gaps = None

        db_conn.format_and_save_gaps(gaps, 5)
        gaps_E = services.get_gaps(
            session,
            db_conn.channel_info.channel_ids["HHE"],
            db_conn.daily_info.contdatainfo_id,
        )
        assert len(gaps_E) == 0, "Incorrect number of gaps on HHE channel"

        gaps_Z = services.get_gaps(
            session,
            db_conn.channel_info.channel_ids["HHZ"],
            db_conn.daily_info.contdatainfo_id,
        )
        assert len(gaps_Z) == 0, "Incorrect number of gaps on HHZ channel"

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

        assert len(selected_dets) == 4, "incorrect number of detections"

    @pytest.fixture
    def db_session_with_P_picks_and_wfs(
        self, db_session_with_P_dldets, contdatainfo_ex
    ):
        session, db_conn = db_session_with_P_dldets
        pick_thresh = 75
        auth = "TEST"
        wf_proc_notes = "TEST DATA"
        seconds_around_pick = 10

        _, metadata = contdatainfo_ex
        cont_data = np.zeros((metadata["npts"], 3))
        samples = int(seconds_around_pick * 100)
        cont_data[1000 - samples : 1000 + samples + 1] = 1
        cont_data[20000 - samples : 20000 + samples + 1] = 2
        cont_data[30000 - samples : 30000 + samples + 1] = 3
        cont_data[(metadata["npts"] - 500) - samples : metadata["npts"]] = 4

        db_conn.save_picks_from_detections(
            pick_thresh=pick_thresh,
            is_p=True,
            auth=auth,
            continuous_data=cont_data,
            wf_filt_low=None,
            wf_filt_high=None,
            wf_proc_notes=wf_proc_notes,
            seconds_around_pick=seconds_around_pick,
            use_pytables=False,
        )

        params = {
            "pick_thresh": pick_thresh,
            "auth": auth,
            "wf_proc_notes": wf_proc_notes,
            "seconds_around_pick": seconds_around_pick,
            "samples": samples,
        }

        return session, db_conn, params

    def test_save_picks_from_detections(self, db_session_with_P_picks_and_wfs):
        session, db_conn, params = db_session_with_P_picks_and_wfs

        picks = services.get_picks(session, db_conn.station_id, "HH", phase="P")
        assert len(picks) == 3, "incorrect number of picks"

        for pick in picks:
            det = session.get(tables.DLDetection, pick.detid)
            assert det.height > params["pick_thresh"]
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

            assert det.sample in [1000, 30000, 8639500], "incorrect dets saved"
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
            elif det.sample == 8639500:
                assert np.all(
                    np.array(wf[0].data) == 4
                ), "invalid data for wf[0] when det.sample == 8639500"
                assert np.all(
                    np.array(wf[1].data) == 4
                ), "invalid data for wf[1] when det.sample == 8639500"
                assert np.all(
                    np.array(wf[2].data) == 4
                ), "invalid data for wf[2] when det.sample == 8639500"

            assert wf[0].start == pick.ptime - timedelta(
                seconds=params["seconds_around_pick"]
            ), "invalid start for wf[0]"
            assert wf[1].start == pick.ptime - timedelta(
                seconds=params["seconds_around_pick"]
            ), "invalid start for wf[1]"
            assert wf[2].start == pick.ptime - timedelta(
                seconds=params["seconds_around_pick"]
            ), "invalid start for wf[2]"
            assert wf[0].filt_low is None
            assert wf[1].filt_low is None
            assert wf[2].filt_low is None
            assert wf[0].filt_high is None
            assert wf[1].filt_high is None
            assert wf[2].filt_high is None

            if det.sample in [1000, 30000]:
                assert (
                    len(wf[0].data) == params["samples"] * 2 + 1
                ), "invalid data length for wf[0]"
                assert (
                    len(wf[1].data) == params["samples"] * 2 + 1
                ), "invalid data length for wf[1]"
                assert (
                    len(wf[2].data) == params["samples"] * 2 + 1
                ), "invalid data length for wf[2]"

                assert wf[0].end == pick.ptime + timedelta(
                    seconds=params["seconds_around_pick"] + 0.01
                ), "invalid end for wf[0]"
                assert wf[1].end == pick.ptime + timedelta(
                    seconds=params["seconds_around_pick"] + 0.01
                ), "invalid end for wf[1]"
                assert wf[2].end == pick.ptime + timedelta(
                    seconds=params["seconds_around_pick"] + 0.01
                ), "invalid end for wf[2]"
            else:
                assert (
                    len(wf[0].data) == 1500
                ), "invalid data length for wf[0] for pick at 8639500"
                assert (
                    len(wf[1].data) == 1500
                ), "invalid data length for wf[1] for pick at 8639500"
                assert (
                    len(wf[2].data) == 1500
                ), "invalid data length for wf[2] for pick at 8639500"

                assert wf[0].end == pick.ptime + timedelta(
                    seconds=(500 * 0.01)
                ), "invalid end for wf[0]"
                assert wf[1].end == pick.ptime + timedelta(
                    seconds=(500 * 0.01)
                ), "invalid end for wf[1]"
                assert wf[2].end == pick.ptime + timedelta(
                    seconds=(500 * 0.01)
                ), "invalid end for wf[2]"

    def test_save_picks_from_detections_handle_append_previous(
        self, db_session_with_P_picks_and_wfs, contdatainfo_ex
    ):
        session, db_conn, params = db_session_with_P_picks_and_wfs
        date, metadata = contdatainfo_ex
        date = date + timedelta(days=1)
        metadata["original_starttime"] += timedelta(days=1)
        metadata["original_endtime"] += timedelta(days=1)
        metadata["previous_appended"] = True
        metadata["starttime"] += timedelta(days=1)
        metadata["starttime"] += -timedelta(seconds=10)
        metadata["npts"] += 10 * metadata["sampling_rate"]
        # print(date, metadata)

        # Updat the info to move to the next date
        db_conn.save_data_info(date, metadata)

        # Make a new detection that is in the previous day's data
        ids = db_conn.get_dldet_fk_ids(is_p=True)
        det = {"sample": 502, "height": 90, "width": 20, "phase": "P"}
        det["data_id"] = ids["data"]
        det["method_id"] = ids["method"]
        det["inference_id"] = None
        db_conn.save_detections([det])
        inserted_dets = services.get_dldetections(
            session, ids["data"], ids["method"], 0.0
        )
        assert len(inserted_dets) == 1, "incorrect number of dets inserted"
        # print("det", det)
        # print(inserted_dets[0].time)

        # close_picks = services.get_picks(
        #                 session,
        #                 db_conn.station_id,
        #                 db_conn.seed_code,
        #                 "P",
        #                 min_time=inserted_dets[0].time - timedelta(seconds=0.1),
        #                 max_time=inserted_dets[0].time + timedelta(seconds=0.1),
        #             )
        # print(close_picks)
        # assert len(close_picks) == 1, "incorrect number of close picks"

        cont_data = np.zeros((int(metadata["npts"]), 3))
        samples = int(params["seconds_around_pick"] * 100)
        cont_data[0 : 502 + samples + 1] = 5

        db_conn.save_picks_from_detections(
            pick_thresh=params["pick_thresh"],
            is_p=True,
            auth=params["auth"],
            continuous_data=cont_data,
            wf_filt_low=None,
            wf_filt_high=None,
            wf_proc_notes=params["wf_proc_notes"],
            seconds_around_pick=params["seconds_around_pick"],
            use_pytables=False,
        )

        picks = services.get_picks(session, db_conn.station_id, "HH", phase="P")
        assert len(picks) == 3, "incorrect number of total picks"
        pick_of_interest = services.get_picks(
            session, db_conn.station_id, "HH", phase="P", min_time=metadata["starttime"]
        )
        assert (
            len(pick_of_interest) == 1
        ), "incorrect number of picks on previous part of data"
        pick_of_interest = pick_of_interest[0]
        assert pick_of_interest.ptime == metadata["starttime"] + timedelta(
            seconds=(502 * metadata["dt"])
        ), "incorrect pick time"
        assert pick_of_interest.detid == inserted_dets[0].id, "incorrect detection id"

        wf = services.get_waveforms(session, pick_of_interest.id)
        assert len(wf) == 3, "invalid wf size"

        assert np.all(np.array(wf[0].data) == 5), "invalid data for wf[0]"
        assert np.all(np.array(wf[1].data) == 5), "invalid data for wf[1]"
        assert np.all(np.array(wf[2].data) == 5), "invalid data for wf[2]"

        assert wf[0].start == metadata["starttime"], "invalid start for wf[0]"
        assert wf[1].start == metadata["starttime"], "invalid start for wf[1]"
        assert wf[2].start == metadata["starttime"], "invalid start for wf[2]"
        assert wf[0].filt_low is None
        assert wf[1].filt_low is None
        assert wf[2].filt_low is None
        assert wf[0].filt_high is None
        assert wf[1].filt_high is None
        assert wf[2].filt_high is None

        # 502 samples before, 1000 + 1 samples after
        assert len(wf[0].data) == 1503, "invalid data length for wf[0]"
        assert len(wf[1].data) == 1503, "invalid data length for wf[1]"
        assert len(wf[2].data) == 1503, "invalid data length for wf[2]"

        assert wf[0].end == pick_of_interest.ptime + timedelta(
            seconds=10 + 0.01
        ), "invalid end for wf[0]"
        assert wf[1].end == pick_of_interest.ptime + timedelta(
            seconds=10 + 0.01
        ), "invalid end for wf[1]"
        assert wf[2].end == pick_of_interest.ptime + timedelta(
            seconds=10 + 0.01
        ), "invalid end for wf[2]"

    @pytest.fixture
    def db_session_with_S_dldets(
        self, db_session_with_saved_contdatainfo, detections_ex
    ):
        session, db_conn = db_session_with_saved_contdatainfo
        det_list = detections_ex
        ids = db_conn.get_dldet_fk_ids(is_p=False)
        for det in det_list:
            det["phase"] = "S"
            det["data_id"] = ids["data"]
            det["method_id"] = ids["method"]

        db_conn.save_detections(det_list)

        return session, db_conn

    @pytest.fixture
    def db_session_with_S_picks_and_wfs(
        self, db_session_with_S_dldets, contdatainfo_ex
    ):
        session, db_conn = db_session_with_S_dldets
        pick_thresh = 75
        auth = "TEST"
        wf_proc_notes = "TEST DATA"
        seconds_around_pick = 10

        _, metadata = contdatainfo_ex
        cont_data = np.zeros((metadata["npts"], 3))
        samples = int(seconds_around_pick * 100)
        cont_data[1000 - samples : 1000 + samples + 1] = 1
        cont_data[20000 - samples : 20000 + samples + 1] = 2
        cont_data[30000 - samples : 30000 + samples + 1] = 3
        cont_data[(metadata["npts"] - 500) - samples : metadata["npts"]] = 4

        db_conn.save_picks_from_detections(
            pick_thresh=pick_thresh,
            is_p=False,
            auth=auth,
            continuous_data=cont_data,
            wf_filt_low=None,
            wf_filt_high=None,
            wf_proc_notes=wf_proc_notes,
            seconds_around_pick=seconds_around_pick,
            use_pytables=False,
        )

        params = {
            "pick_thresh": pick_thresh,
            "auth": auth,
            "wf_proc_notes": wf_proc_notes,
            "seconds_around_pick": seconds_around_pick,
            "samples": samples,
        }

        return session, db_conn, params

    def test_save_picks_from_detections_handle_append_previous_S(
        self, db_session_with_S_picks_and_wfs, contdatainfo_ex
    ):
        session, db_conn, params = db_session_with_S_picks_and_wfs
        date, metadata = contdatainfo_ex
        date = date + timedelta(days=1)
        metadata["original_starttime"] += timedelta(days=1)
        metadata["original_endtime"] += timedelta(days=1)
        metadata["previous_appended"] = True
        metadata["starttime"] += timedelta(days=1)
        metadata["starttime"] += -timedelta(seconds=10)
        metadata["npts"] += 10 * metadata["sampling_rate"]
        # print(date, metadata)

        # Updat the info to move to the next date
        db_conn.save_data_info(date, metadata)

        # Make a new detection that is in the previous day's data
        ids = db_conn.get_dldet_fk_ids(is_p=False)
        det = {"sample": 502, "height": 90, "width": 20, "phase": "S"}
        det["data_id"] = ids["data"]
        det["method_id"] = ids["method"]
        det["inference_id"] = None
        db_conn.save_detections([det])
        inserted_dets = services.get_dldetections(
            session, ids["data"], ids["method"], 0.0
        )
        assert len(inserted_dets) == 1, "incorrect number of dets inserted"

        cont_data = np.zeros((int(metadata["npts"]), 3))
        samples = int(params["seconds_around_pick"] * 100)
        cont_data[0 : 502 + samples + 1] = 5

        db_conn.save_picks_from_detections(
            pick_thresh=params["pick_thresh"],
            is_p=False,
            auth=params["auth"],
            continuous_data=cont_data,
            wf_filt_low=None,
            wf_filt_high=None,
            wf_proc_notes=params["wf_proc_notes"],
            seconds_around_pick=params["seconds_around_pick"],
            use_pytables=False,
        )

        picks = services.get_picks(session, db_conn.station_id, "HH", phase="S")
        assert len(picks) == 3, "incorrect number of total picks"
        pick_of_interest = services.get_picks(
            session, db_conn.station_id, "HH", phase="S", min_time=metadata["starttime"]
        )
        assert (
            len(pick_of_interest) == 1
        ), "incorrect number of picks on previous part of data"
        pick_of_interest = pick_of_interest[0]
        assert pick_of_interest.ptime == metadata["starttime"] + timedelta(
            seconds=(502 * metadata["dt"])
        ), "incorrect pick time"
        assert pick_of_interest.detid == inserted_dets[0].id, "incorrect detection id"

        wf = services.get_waveforms(session, pick_of_interest.id)
        assert len(wf) == 3, "invalid wf size"

        assert np.all(np.array(wf[0].data) == 5), "invalid data for wf[0]"
        assert np.all(np.array(wf[1].data) == 5), "invalid data for wf[1]"
        assert np.all(np.array(wf[2].data) == 5), "invalid data for wf[2]"

        assert wf[0].start == metadata["starttime"], "invalid start for wf[0]"
        assert wf[1].start == metadata["starttime"], "invalid start for wf[1]"
        assert wf[2].start == metadata["starttime"], "invalid start for wf[2]"
        assert wf[0].filt_low is None
        assert wf[1].filt_low is None
        assert wf[2].filt_low is None
        assert wf[0].filt_high is None
        assert wf[1].filt_high is None
        assert wf[2].filt_high is None

        # 502 samples before, 1000 + 1 samples after
        assert len(wf[0].data) == 1503, "invalid data length for wf[0]"
        assert len(wf[1].data) == 1503, "invalid data length for wf[1]"
        assert len(wf[2].data) == 1503, "invalid data length for wf[2]"

        assert wf[0].end == pick_of_interest.ptime + timedelta(
            seconds=10 + 0.01
        ), "invalid end for wf[0]"
        assert wf[1].end == pick_of_interest.ptime + timedelta(
            seconds=10 + 0.01
        ), "invalid end for wf[1]"
        assert wf[2].end == pick_of_interest.ptime + timedelta(
            seconds=10 + 0.01
        ), "invalid end for wf[2]"

    def test_open_dldetection_output_storage(
        self, db_session_with_saved_contdatainfo, mock_pytables_config
    ):
        session, db_conn = db_session_with_saved_contdatainfo
        detmethod_id = db_conn.p_detection_method_id
        try:
            detout_storage = db_conn._open_dldetection_output_storage(
                1000, "P", detmethod_id
            )
            file_name = detout_storage.file_name
            # Check that the filename is as would be expected
            assert (
                file_name == f"WY.YNR..HH.P.3C.detmethod{detmethod_id:02d}.h5"
            ), "file name is not as expected"
            # Check that the file was created
            assert os.path.exists(detout_storage.file_path), "the file was not created"
            # Check that the file is set to open
            assert detout_storage._is_open, "the file is not registered as open"
            table = detout_storage.table
            # Check table info
            assert table.name == "dldetector_output", "the table name is incorrect"
            assert table.title == "DL detector output", "the table title is incorrect"
            # Check table attributes
            assert table.attrs.sta == "YNR", "the table sta attr is incorrect"
            assert (
                table.attrs.seed_code == "HH"
            ), "the table seed_code attr is incorrect"
            assert table.attrs.ncomps == 3, "the table ncomps attr is incorrect"
            assert table.attrs.phase == "P", "the table phase attr is incorrect"
            assert (
                table.attrs.det_method_id == detmethod_id
            ), "the table det_method_id attr is incorrect"
            assert (
                table.attrs.expected_array_length == 1000
            ), "the table expected_array_length attr is incorrect"
        finally:
            # Clean up
            if detout_storage is not None:
                detout_storage.close()
                os.remove(detout_storage.file_path)
                assert not os.path.exists(
                    detout_storage.file_path
                ), "the file was not removed"

    def test_save_detection_output(
        self, db_session_with_saved_contdatainfo, mock_pytables_config
    ):
        session, db_conn = db_session_with_saved_contdatainfo
        detmethod_id = db_conn.p_detection_method_id
        try:
            detout_storage = db_conn._open_dldetection_output_storage(
                1000, "P", detmethod_id
            )

            data = (np.random.rand(1000) * 100).astype(np.uint8)
            detout_id = db_conn._save_detection_output(
                detout_storage, data, detmethod_id
            )

            assert detout_id is not None, "detector_output id is not defined"
            detout = session.get(tables.DLDetectorOutput, detout_id)
            assert (
                detout.data_id == db_conn.daily_info.contdatainfo_id
            ), "incorrect data_id"
            assert detout.method_id == detmethod_id, "incorrect method_id"
            assert (
                detout.hdf_file == detout_storage.file_name
            ), "incorrect filename stored in db"

            assert (
                detout_storage.table.nrows == 1
            ), "incorrect number of rows in the table"
            assert detout_storage.table[0]["id"] == detout.id, "incorrect id in pytable"
            assert np.array_equal(
                detout_storage.table[0]["data"], data
            ), "data incorrect"
            assert not detout_storage._in_transaction, "transaction wasn't closed"
        finally:
            # Clean up
            if detout_storage is not None:
                detout_storage.close()
                os.remove(detout_storage.file_path)
                assert not os.path.exists(
                    detout_storage.file_path
                ), "the file was not removed"

    def test_save_P_post_probs(
        self, db_session_with_saved_contdatainfo, mock_pytables_config
    ):
        session, db_conn = db_session_with_saved_contdatainfo
        data = (np.random.rand(1000) * 100).astype(np.uint8)
        try:
            assert db_conn.daily_info.dldet_output_id_P is None
            db_conn.save_P_post_probs(data, 1000)
            assert (
                db_conn.daily_info.dldet_output_id_P is not None
            ), "dldetector_output.id not stored"
            assert (
                db_conn.daily_info.dldet_output_id_S is None
            ), "S detout id shouldl be None"
            detout = session.get(
                tables.DLDetectorOutput, db_conn.daily_info.dldet_output_id_P
            )
            assert (
                detout.method_id == db_conn.p_detection_method_id
            ), "incorrect method id"
            assert detout is not None, "row not found in db"
            assert db_conn.detout_storage_P.table.nrows == 1, "data not inserted"
            assert np.array_equal(
                db_conn.detout_storage_P.table[0]["data"], data
            ), "data incorrect"
            assert (
                db_conn.detout_storage_P.file_name
                == f"WY.YNR..HH.P.3C.detmethod{detout.method_id:02d}.h5"
            ), "file name is not as expected"
        finally:
            # Clean up
            if db_conn.detout_storage_P is not None:
                db_conn.detout_storage_P.close()
                os.remove(db_conn.detout_storage_P.file_path)
                assert not os.path.exists(
                    db_conn.detout_storage_P.file_path
                ), "the file was not removed"

    def test_save_S_post_probs(
        self, db_session_with_saved_contdatainfo, mock_pytables_config
    ):
        session, db_conn = db_session_with_saved_contdatainfo
        data = (np.random.rand(1000) * 100).astype(np.uint8)
        try:
            assert db_conn.daily_info.dldet_output_id_S is None
            db_conn.save_S_post_probs(data, 1000)
            assert (
                db_conn.daily_info.dldet_output_id_S is not None
            ), "dldetector_output.id not stored"
            assert (
                db_conn.daily_info.dldet_output_id_P is None
            ), "P detout id should be None"
            detout = session.get(
                tables.DLDetectorOutput, db_conn.daily_info.dldet_output_id_S
            )
            assert (
                detout.method_id == db_conn.s_detection_method_id
            ), "incorrect method id"
            assert detout is not None, "row not found in db"
            assert db_conn.detout_storage_S.table.nrows == 1, "data not inserted"
            assert np.array_equal(
                db_conn.detout_storage_S.table[0]["data"], data
            ), "data incorrect"
            assert (
                db_conn.detout_storage_S.file_name
                == f"WY.YNR..HH.S.3C.detmethod{detout.method_id:02d}.h5"
            ), "file name is not as expected"
        finally:
            # Clean up
            if db_conn.detout_storage_S is not None:
                db_conn.detout_storage_S.close()
                os.remove(db_conn.detout_storage_S.file_path)
                assert not os.path.exists(
                    db_conn.detout_storage_S.file_path
                ), "the file was not removed"

    @pytest.fixture
    def db_session_with_P_picks_and_wfs_pytables(
        self, db_session_with_P_dldets, contdatainfo_ex, mock_pytables_config
    ):
        session, db_conn = db_session_with_P_dldets
        pick_thresh = 75
        auth = "TEST"
        wf_proc_notes = "TEST DATA"
        seconds_around_pick = 10

        _, metadata = contdatainfo_ex
        cont_data = np.zeros((metadata["npts"], 3))
        samples = int(seconds_around_pick * 100)
        cont_data[1000 - samples : 1000 + samples + 1] = 1
        cont_data[20000 - samples : 20000 + samples + 1] = 2
        cont_data[30000 - samples : 30000 + samples + 1] = 3
        cont_data[(metadata["npts"] - 500) - samples : metadata["npts"]] = 4

        db_conn.save_picks_from_detections(
            pick_thresh=pick_thresh,
            is_p=True,
            auth=auth,
            continuous_data=cont_data,
            wf_filt_low=None,
            wf_filt_high=None,
            wf_proc_notes=wf_proc_notes,
            seconds_around_pick=seconds_around_pick,
            use_pytables=True,
        )

        params = {
            "pick_thresh": pick_thresh,
            "auth": auth,
            "wf_proc_notes": wf_proc_notes,
            "seconds_around_pick": seconds_around_pick,
            "samples": samples,
        }

        return session, db_conn, params

    def test_save_picks_from_detections_pytables(
        self, db_session_with_P_picks_and_wfs_pytables
    ):
        session, db_conn, params = db_session_with_P_picks_and_wfs_pytables

        picks = services.get_picks(session, db_conn.station_id, "HH", phase="P")
        assert len(picks) == 3, "incorrect number of picks"
        try:
            for pick in picks:
                det = session.get(tables.DLDetection, pick.detid)
                assert det.height > params["pick_thresh"]
                contdatainfo = session.get(tables.DailyContDataInfo, det.data_id)

                assert (
                    pick.ptime
                    - timedelta(seconds=(det.sample / contdatainfo.samp_rate))
                    == contdatainfo.proc_start
                ), "invalid pick time"

                assert pick.auth == "TEST", "invalid author"
                assert pick.phase == "P"
                assert pick.chan_pref == "HH"
                assert pick.sta_id == db_conn.station_id
                assert pick.snr is None
                assert pick.amp is None

                wf_infos = services.get_waveform_infos(session, pick.id)
                assert len(wf_infos) == 3, "invalid wf_info size"

                wfs = []
                for wf_info in wf_infos:
                    row = db_conn.waveform_storage_dict_P[wf_info.chan_id].select_row(
                        wf_info.id
                    )
                    wfs.append(row["data"][row["start_ind"] : row["end_ind"]])

                assert det.sample in [1000, 30000, 8639500], "incorrect dets saved"
                if det.sample == 1000:
                    assert np.all(
                        wfs[0] == 1
                    ), "invalid data for wf[0] when det.sample == 1000"
                    assert np.all(
                        wfs[1] == 1
                    ), "invalid data for wf[1] when det.sample == 1000"
                    assert np.all(
                        wfs[2] == 1
                    ), "invalid data for wf[2] when det.sample == 1000"
                elif det.sample == 30000:
                    assert np.all(
                        wfs[0] == 3
                    ), "invalid data for wf[0] when det.sample == 30000"
                    assert np.all(
                        wfs[1] == 3
                    ), "invalid data for wf[1] when det.sample == 30000"
                    assert np.all(
                        wfs[2] == 3
                    ), "invalid data for wf[2] when det.sample == 30000"
                elif det.sample == 8639500:
                    print(wfs)
                    assert np.all(
                        wfs[0] == 4
                    ), "invalid data for wf[0] when det.sample == 8639500"
                    assert np.all(
                        wfs[1] == 4
                    ), "invalid data for wf[1] when det.sample == 8639500"
                    assert np.all(
                        wfs[2] == 4
                    ), "invalid data for wf[2] when det.sample == 8639500"

                assert wf_infos[0].start == pick.ptime - timedelta(
                    seconds=params["seconds_around_pick"]
                ), "invalid start for wf[0]"
                assert wf_infos[1].start == pick.ptime - timedelta(
                    seconds=params["seconds_around_pick"]
                ), "invalid start for wf[1]"
                assert wf_infos[2].start == pick.ptime - timedelta(
                    seconds=params["seconds_around_pick"]
                ), "invalid start for wf[2]"
                assert wf_infos[0].filt_low is None
                assert wf_infos[1].filt_low is None
                assert wf_infos[2].filt_low is None
                assert wf_infos[0].filt_high is None
                assert wf_infos[1].filt_high is None
                assert wf_infos[2].filt_high is None

                if det.sample in [1000, 30000]:
                    assert (
                        len(wfs[0]) == params["samples"] * 2 + 1
                    ), "invalid data length for wf[0]"
                    assert (
                        len(wfs[1]) == params["samples"] * 2 + 1
                    ), "invalid data length for wf[1]"
                    assert (
                        len(wfs[2]) == params["samples"] * 2 + 1
                    ), "invalid data length for wf[2]"

                    assert wf_infos[0].end == pick.ptime + timedelta(
                        seconds=params["seconds_around_pick"] + 0.01
                    ), "invalid end for wf[0]"
                    assert wf_infos[1].end == pick.ptime + timedelta(
                        seconds=params["seconds_around_pick"] + 0.01
                    ), "invalid end for wf[1]"
                    assert wf_infos[2].end == pick.ptime + timedelta(
                        seconds=params["seconds_around_pick"] + 0.01
                    ), "invalid end for wf[2]"
                else:
                    assert (
                        len(wfs[0]) == 1500
                    ), "invalid data length for wf[0] for pick at 8639500"
                    assert (
                        len(wfs[1]) == 1500
                    ), "invalid data length for wf[1] for pick at 8639500"
                    assert (
                        len(wfs[2]) == 1500
                    ), "invalid data length for wf[2] for pick at 8639500"

                    assert wf_infos[0].end == pick.ptime + timedelta(
                        seconds=(500 * 0.01)
                    ), "invalid end for wf[0]"
                    assert wf_infos[1].end == pick.ptime + timedelta(
                        seconds=(500 * 0.01)
                    ), "invalid end for wf[1]"
                    assert wf_infos[2].end == pick.ptime + timedelta(
                        seconds=(500 * 0.01)
                    ), "invalid end for wf[2]"
        finally:
            # Clean up
            db_conn.close_open_pytables()
            if db_conn.waveform_storage_dict_P is not None:
                for _, stor in db_conn.waveform_storage_dict_P.items():
                    os.remove(stor.file_path)
                    assert not os.path.exists(
                        stor.file_path
                    ), "the file was not removed"

    @pytest.fixture
    def db_session_with_S_picks_and_wfs_pytables(
        self, db_session_with_S_dldets, contdatainfo_ex, mock_pytables_config
    ):
        session, db_conn = db_session_with_S_dldets
        pick_thresh = 75
        auth = "TEST"
        wf_proc_notes = "TEST DATA"
        seconds_around_pick = 10

        _, metadata = contdatainfo_ex
        cont_data = np.zeros((metadata["npts"], 3))
        samples = int(seconds_around_pick * 100)
        cont_data[1000 - samples : 1000 + samples + 1] = 1
        cont_data[20000 - samples : 20000 + samples + 1] = 2
        cont_data[30000 - samples : 30000 + samples + 1] = 3
        cont_data[(metadata["npts"] - 500) - samples : metadata["npts"]] = 4

        db_conn.save_picks_from_detections(
            pick_thresh=pick_thresh,
            is_p=False,
            auth=auth,
            continuous_data=cont_data,
            wf_filt_low=None,
            wf_filt_high=None,
            wf_proc_notes=wf_proc_notes,
            seconds_around_pick=seconds_around_pick,
            use_pytables=True,
        )

        params = {
            "pick_thresh": pick_thresh,
            "auth": auth,
            "wf_proc_notes": wf_proc_notes,
            "seconds_around_pick": seconds_around_pick,
            "samples": samples,
        }

        return session, db_conn, params

    def test_save_picks_from_detections_pytables_S(
        self, db_session_with_S_picks_and_wfs_pytables
    ):
        session, db_conn, params = db_session_with_S_picks_and_wfs_pytables

        picks = services.get_picks(session, db_conn.station_id, "HH", phase="S")
        assert len(picks) == 3, "incorrect number of picks"
        try:
            for pick in picks:
                det = session.get(tables.DLDetection, pick.detid)
                assert det.height > params["pick_thresh"]
                contdatainfo = session.get(tables.DailyContDataInfo, det.data_id)

                assert (
                    pick.ptime
                    - timedelta(seconds=(det.sample / contdatainfo.samp_rate))
                    == contdatainfo.proc_start
                ), "invalid pick time"

                assert pick.auth == "TEST", "invalid author"
                assert pick.phase == "S"
                assert pick.chan_pref == "HH"
                assert pick.sta_id == db_conn.station_id
                assert pick.snr is None
                assert pick.amp is None

                wf_infos = services.get_waveform_infos(session, pick.id)
                assert len(wf_infos) == 3, "invalid wf_info size"

                wfs = []
                for wf_info in wf_infos:
                    row = db_conn.waveform_storage_dict_S[wf_info.chan_id].select_row(
                        wf_info.id
                    )
                    wfs.append(row["data"][row["start_ind"] : row["end_ind"]])

                assert det.sample in [1000, 30000, 8639500], "incorrect dets saved"
                if det.sample == 1000:
                    assert np.all(
                        wfs[0] == 1
                    ), "invalid data for wf[0] when det.sample == 1000"
                    assert np.all(
                        wfs[1] == 1
                    ), "invalid data for wf[1] when det.sample == 1000"
                    assert np.all(
                        wfs[2] == 1
                    ), "invalid data for wf[2] when det.sample == 1000"
                elif det.sample == 30000:
                    assert np.all(
                        wfs[0] == 3
                    ), "invalid data for wf[0] when det.sample == 30000"
                    assert np.all(
                        wfs[1] == 3
                    ), "invalid data for wf[1] when det.sample == 30000"
                    assert np.all(
                        wfs[2] == 3
                    ), "invalid data for wf[2] when det.sample == 30000"
                elif det.sample == 8639500:
                    print(wfs)
                    assert np.all(
                        wfs[0] == 4
                    ), "invalid data for wf[0] when det.sample == 8639500"
                    assert np.all(
                        wfs[1] == 4
                    ), "invalid data for wf[1] when det.sample == 8639500"
                    assert np.all(
                        wfs[2] == 4
                    ), "invalid data for wf[2] when det.sample == 8639500"

                assert wf_infos[0].start == pick.ptime - timedelta(
                    seconds=params["seconds_around_pick"]
                ), "invalid start for wf[0]"
                assert wf_infos[1].start == pick.ptime - timedelta(
                    seconds=params["seconds_around_pick"]
                ), "invalid start for wf[1]"
                assert wf_infos[2].start == pick.ptime - timedelta(
                    seconds=params["seconds_around_pick"]
                ), "invalid start for wf[2]"
                assert wf_infos[0].filt_low is None
                assert wf_infos[1].filt_low is None
                assert wf_infos[2].filt_low is None
                assert wf_infos[0].filt_high is None
                assert wf_infos[1].filt_high is None
                assert wf_infos[2].filt_high is None

                if det.sample in [1000, 30000]:
                    assert (
                        len(wfs[0]) == params["samples"] * 2 + 1
                    ), "invalid data length for wf[0]"
                    assert (
                        len(wfs[1]) == params["samples"] * 2 + 1
                    ), "invalid data length for wf[1]"
                    assert (
                        len(wfs[2]) == params["samples"] * 2 + 1
                    ), "invalid data length for wf[2]"

                    assert wf_infos[0].end == pick.ptime + timedelta(
                        seconds=params["seconds_around_pick"] + 0.01
                    ), "invalid end for wf[0]"
                    assert wf_infos[1].end == pick.ptime + timedelta(
                        seconds=params["seconds_around_pick"] + 0.01
                    ), "invalid end for wf[1]"
                    assert wf_infos[2].end == pick.ptime + timedelta(
                        seconds=params["seconds_around_pick"] + 0.01
                    ), "invalid end for wf[2]"
                else:
                    assert (
                        len(wfs[0]) == 1500
                    ), "invalid data length for wf[0] for pick at 8639500"
                    assert (
                        len(wfs[1]) == 1500
                    ), "invalid data length for wf[1] for pick at 8639500"
                    assert (
                        len(wfs[2]) == 1500
                    ), "invalid data length for wf[2] for pick at 8639500"

                    assert wf_infos[0].end == pick.ptime + timedelta(
                        seconds=(500 * 0.01)
                    ), "invalid end for wf[0]"
                    assert wf_infos[1].end == pick.ptime + timedelta(
                        seconds=(500 * 0.01)
                    ), "invalid end for wf[1]"
                    assert wf_infos[2].end == pick.ptime + timedelta(
                        seconds=(500 * 0.01)
                    ), "invalid end for wf[2]"
        finally:
            # Clean up
            db_conn.close_open_pytables()
            if db_conn.waveform_storage_dict_S is not None:
                for _, stor in db_conn.waveform_storage_dict_S.items():
                    os.remove(stor.file_path)
                    assert not os.path.exists(
                        stor.file_path
                    ), "the file was not removed"

    def test_save_picks_from_detections_handle_append_previous_pytables(
        self, db_session_with_P_picks_and_wfs_pytables, contdatainfo_ex
    ):
        try:
            session, db_conn, params = db_session_with_P_picks_and_wfs_pytables
            date, metadata = contdatainfo_ex
            date = date + timedelta(days=1)
            metadata["original_starttime"] += timedelta(days=1)
            metadata["original_endtime"] += timedelta(days=1)
            metadata["previous_appended"] = True
            metadata["starttime"] += timedelta(days=1)
            metadata["starttime"] += -timedelta(seconds=10)
            metadata["npts"] += 10 * metadata["sampling_rate"]
            # print(date, metadata)

            # Updat the info to move to the next date
            db_conn.save_data_info(date, metadata)

            # Make a new detection that is in the previous day's data
            ids = db_conn.get_dldet_fk_ids(is_p=True)
            det = {"sample": 502, "height": 90, "width": 20, "phase": "P"}
            det["data_id"] = ids["data"]
            det["method_id"] = ids["method"]
            det["inference_id"] = None
            db_conn.save_detections([det])
            inserted_dets = services.get_dldetections(
                session, ids["data"], ids["method"], 0.0
            )
            assert len(inserted_dets) == 1, "incorrect number of dets inserted"

            cont_data = np.zeros((int(metadata["npts"]), 3))
            samples = int(params["seconds_around_pick"] * 100)
            cont_data[0 : 502 + samples + 1] = 5

            db_conn.save_picks_from_detections(
                pick_thresh=params["pick_thresh"],
                is_p=True,
                auth=params["auth"],
                continuous_data=cont_data,
                wf_filt_low=None,
                wf_filt_high=None,
                wf_proc_notes=params["wf_proc_notes"],
                seconds_around_pick=params["seconds_around_pick"],
                use_pytables=True,
            )

            picks = services.get_picks(session, db_conn.station_id, "HH", phase="P")
            assert len(picks) == 3, "incorrect number of total picks"
            pick_of_interest = services.get_picks(
                session,
                db_conn.station_id,
                "HH",
                phase="P",
                min_time=metadata["starttime"],
            )
            assert (
                len(pick_of_interest) == 1
            ), "incorrect number of picks on previous part of data"
            pick_of_interest = pick_of_interest[0]
            assert pick_of_interest.ptime == metadata["starttime"] + timedelta(
                seconds=(502 * metadata["dt"])
            ), "incorrect pick time"
            assert (
                pick_of_interest.detid == inserted_dets[0].id
            ), "incorrect detection id"

            wf_infos = services.get_waveform_infos(session, pick_of_interest.id)
            assert len(wf_infos) == 3, "invalid wf_info size"

            wfs = []
            for wf_info in wf_infos:
                row = db_conn.waveform_storage_dict_P[wf_info.chan_id].select_row(
                    wf_info.id
                )
                wfs.append(row["data"][row["start_ind"] : row["end_ind"]])
            assert len(wfs) == 3, "invalid wf size"

            assert np.all(wfs[0] == 5), "invalid data for wf[0]"
            assert np.all(wfs[1] == 5), "invalid data for wf[1]"
            assert np.all(wfs[2] == 5), "invalid data for wf[2]"

            assert wf_infos[0].start == metadata["starttime"], "invalid start for wf[0]"
            assert wf_infos[1].start == metadata["starttime"], "invalid start for wf[1]"
            assert wf_infos[2].start == metadata["starttime"], "invalid start for wf[2]"
            assert wf_infos[0].filt_low is None
            assert wf_infos[1].filt_low is None
            assert wf_infos[2].filt_low is None
            assert wf_infos[0].filt_high is None
            assert wf_infos[1].filt_high is None
            assert wf_infos[2].filt_high is None

            # 502 samples before, 1000 + 1 samples after
            assert len(wfs[0]) == 1503, "invalid data length for wf[0]"
            assert len(wfs[1]) == 1503, "invalid data length for wf[1]"
            assert len(wfs[2]) == 1503, "invalid data length for wf[2]"

            assert wf_infos[0].end == pick_of_interest.ptime + timedelta(
                seconds=10 + 0.01
            ), "invalid end for wf[0]"
            assert wf_infos[1].end == pick_of_interest.ptime + timedelta(
                seconds=10 + 0.01
            ), "invalid end for wf[1]"
            assert wf_infos[2].end == pick_of_interest.ptime + timedelta(
                seconds=10 + 0.01
            ), "invalid end for wf[2]"
        finally:
            # Clean up
            db_conn.close_open_pytables()
            if db_conn.waveform_storage_dict_P is not None:
                for _, stor in db_conn.waveform_storage_dict_P.items():
                    os.remove(stor.file_path)
                    assert not os.path.exists(
                        stor.file_path
                    ), "the file was not removed"

    def test_save_picks_from_detections_handle_append_previous_pytables_S(
        self, db_session_with_S_picks_and_wfs_pytables, contdatainfo_ex
    ):
        try:
            session, db_conn, params = db_session_with_S_picks_and_wfs_pytables
            date, metadata = contdatainfo_ex
            date = date + timedelta(days=1)
            metadata["original_starttime"] += timedelta(days=1)
            metadata["original_endtime"] += timedelta(days=1)
            metadata["previous_appended"] = True
            metadata["starttime"] += timedelta(days=1)
            metadata["starttime"] += -timedelta(seconds=10)
            metadata["npts"] += 10 * metadata["sampling_rate"]
            # print(date, metadata)

            # Updat the info to move to the next date
            db_conn.save_data_info(date, metadata)

            # Make a new detection that is in the previous day's data
            ids = db_conn.get_dldet_fk_ids(is_p=False)
            det = {"sample": 502, "height": 90, "width": 20, "phase": "S"}
            det["data_id"] = ids["data"]
            det["method_id"] = ids["method"]
            det["inference_id"] = None
            db_conn.save_detections([det])
            inserted_dets = services.get_dldetections(
                session, ids["data"], ids["method"], 0.0
            )
            assert len(inserted_dets) == 1, "incorrect number of dets inserted"

            cont_data = np.zeros((int(metadata["npts"]), 3))
            samples = int(params["seconds_around_pick"] * 100)
            cont_data[0 : 502 + samples + 1] = 5

            db_conn.save_picks_from_detections(
                pick_thresh=params["pick_thresh"],
                is_p=False,
                auth=params["auth"],
                continuous_data=cont_data,
                wf_filt_low=None,
                wf_filt_high=None,
                wf_proc_notes=params["wf_proc_notes"],
                seconds_around_pick=params["seconds_around_pick"],
                use_pytables=True,
            )

            picks = services.get_picks(session, db_conn.station_id, "HH", phase="S")
            assert len(picks) == 3, "incorrect number of total picks"
            pick_of_interest = services.get_picks(
                session,
                db_conn.station_id,
                "HH",
                phase="S",
                min_time=metadata["starttime"],
            )
            assert (
                len(pick_of_interest) == 1
            ), "incorrect number of picks on previous part of data"
            pick_of_interest = pick_of_interest[0]
            assert pick_of_interest.ptime == metadata["starttime"] + timedelta(
                seconds=(502 * metadata["dt"])
            ), "incorrect pick time"
            assert (
                pick_of_interest.detid == inserted_dets[0].id
            ), "incorrect detection id"

            wf_infos = services.get_waveform_infos(session, pick_of_interest.id)
            assert len(wf_infos) == 3, "invalid wf_info size"

            wfs = []
            for wf_info in wf_infos:
                row = db_conn.waveform_storage_dict_S[wf_info.chan_id].select_row(
                    wf_info.id
                )
                wfs.append(row["data"][row["start_ind"] : row["end_ind"]])
            assert len(wfs) == 3, "invalid wf size"

            assert np.all(wfs[0] == 5), "invalid data for wf[0]"
            assert np.all(wfs[1] == 5), "invalid data for wf[1]"
            assert np.all(wfs[2] == 5), "invalid data for wf[2]"

            assert wf_infos[0].start == metadata["starttime"], "invalid start for wf[0]"
            assert wf_infos[1].start == metadata["starttime"], "invalid start for wf[1]"
            assert wf_infos[2].start == metadata["starttime"], "invalid start for wf[2]"
            assert wf_infos[0].filt_low is None
            assert wf_infos[1].filt_low is None
            assert wf_infos[2].filt_low is None
            assert wf_infos[0].filt_high is None
            assert wf_infos[1].filt_high is None
            assert wf_infos[2].filt_high is None

            # 502 samples before, 1000 + 1 samples after
            assert len(wfs[0]) == 1503, "invalid data length for wf[0]"
            assert len(wfs[1]) == 1503, "invalid data length for wf[1]"
            assert len(wfs[2]) == 1503, "invalid data length for wf[2]"

            assert wf_infos[0].end == pick_of_interest.ptime + timedelta(
                seconds=10 + 0.01
            ), "invalid end for wf[0]"
            assert wf_infos[1].end == pick_of_interest.ptime + timedelta(
                seconds=10 + 0.01
            ), "invalid end for wf[1]"
            assert wf_infos[2].end == pick_of_interest.ptime + timedelta(
                seconds=10 + 0.01
            ), "invalid end for wf[2]"
        finally:
            # Clean up
            db_conn.close_open_pytables()
            if db_conn.waveform_storage_dict_S is not None:
                for _, stor in db_conn.waveform_storage_dict_S.items():
                    os.remove(stor.file_path)
                    assert not os.path.exists(
                        stor.file_path
                    ), "the file was not removed"


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
        "det_method_1c_P": {"name": "TEST_1C_P_UNET", "desc": "test for 1C P dets"},
        "det_method_3c_P": {"name": "TEST_3C_P_UNET", "desc": "test for 3C P dets"},
        "det_method_3c_S": {"name": "TEST_3C_S_UNET", "desc": "test for 3C S dets"},
        "p_det_thresh_3c": 50,
        "p_det_thresh_1c": 55,
        "s_det_thresh": 50,
        "p_pick_thresh_3c": 75,
        "p_pick_thresh_1c": 80,
        "s_pick_thresh": 75,
        "wf_seconds_around_pick": 10,
        "pick_author": "SPDL",
        "min_gap_separation_seconds": 5,
        "use_pytables": False,
    },
}


@pytest.fixture
def simple_obspy_gaps_ex():
    gap1 = [
        "Net",
        "Stat",
        "",
        "HHZ",
        UTC("2011-09-10T01:00:00.00"),
        UTC("2011-09-10T02:00:00.00"),
        777,
        777,
    ]

    gap2 = [
        "Net",
        "Stat",
        "",
        "HHZ",
        UTC("2011-09-10T03:00:00.00"),
        UTC("2011-09-10T04:00:00.00"),
        777,
        777,
    ]

    gap3 = [
        "Net",
        "Stat",
        "",
        "HHZ",
        UTC("2011-09-10T05:00:00.00"),
        UTC("2011-09-10T06:00:00.00"),
        777,
        777,
    ]

    return deepcopy([gap1, gap2, gap3])


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
        assert applier.min_gap_sep_seconds == 5

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
        assert applier.p_det_thresh == 55
        assert applier.s_det_thresh is None
        assert applier.p_pick_thresh == 80
        assert applier.s_pick_thresh is None
        assert applier.wf_seconds_around_pick == 10.0
        assert applier.db_pick_author == "SPDL"
        assert applier.min_gap_sep_seconds == 5

    def test_save_daily_results_in_db_1C(
        self, db_session, contdatainfo_ex, simple_obspy_gaps_ex, mock_pytables_config
    ):
        session, _ = db_session
        applier = ApplyDetector(
            1, apply_detector_config, session_factory=lambda: session
        )

        date, metadata = contdatainfo_ex
        gaps = simple_obspy_gaps_ex
        error = None

        continuous_data = np.zeros((metadata["npts"], 1))
        p_post_probs = np.zeros(metadata["npts"])
        p_post_probs[10000] = 90
        p_post_probs[25000] = 85
        p_post_probs[40000] = 80
        p_post_probs[56000] = 56
        p_post_probs[75000] = 45

        applier.db_conn.get_channel_dates(date, "WY", "YNR", "", "HHZ")

        applier.save_daily_results_in_db(
            date, continuous_data, metadata, gaps, error, p_post_probs
        )

        # check the metadata
        assert (
            applier.db_conn.daily_info.date == date
        ), "invalid date in DailyDetectionDBInfo"
        assert (
            applier.db_conn.daily_info.contdatainfo_id is not None
        ), "contdatainfo id not set"
        contdatainfo = session.get(
            tables.DailyContDataInfo, applier.db_conn.daily_info.contdatainfo_id
        )
        assert inspect(contdatainfo).persistent, "contdatainfo not persistent"
        assert contdatainfo is not None, "contdatainfo not set"
        assert contdatainfo.chan_pref == "HHZ", "invalid chan_pref"
        assert contdatainfo.date == date.date(), "contdatainfo date incorrect"
        assert contdatainfo.proc_start == datetime.strptime(
            "2011-09-10T00:00:00.00", datetimeformat
        ), "invalid proc_start"

        # check the gaps
        gaps_Z = services.get_gaps(
            session,
            applier.db_conn.channel_info.channel_ids["HHZ"],
            applier.db_conn.daily_info.contdatainfo_id,
        )
        assert len(gaps_Z) == 3, "Incorrect number of gaps on HHZ channel"

        # Check the Detection Output
        assert applier.db_conn.detout_storage_P is None, "detout_storage_P is set"

        # check the detections
        det_fk_ids = applier.db_conn.get_dldet_fk_ids(is_p=True)
        inserted_dets = services.get_dldetections(
            session, det_fk_ids["data"], det_fk_ids["method"], 0.0, phase="P"
        )
        assert len(inserted_dets) == 4, "incorrect number of detections inserted"

        # check the picks
        picks = services.get_picks(
            session, applier.db_conn.station_id, "HHZ", phase="P"
        )
        assert len(picks) == 3, "incorrect number of picks"

        # check the waveforms
        for pick in picks:
            wf = services.get_waveforms(session, pick.id)
            assert len(wf) == 1, "invalid wf size"

    def test_save_daily_results_in_db_3C(
        self, db_session, contdatainfo_ex, simple_obspy_gaps_ex, mock_pytables_config
    ):
        session, _ = db_session
        applier = ApplyDetector(
            3, apply_detector_config, session_factory=lambda: session
        )

        date, metadata = contdatainfo_ex
        gaps = simple_obspy_gaps_ex
        error = None

        continuous_data = np.zeros((metadata["npts"], 3))
        p_post_probs = np.zeros(metadata["npts"])
        p_post_probs[10000] = 90
        p_post_probs[25000] = 80
        p_post_probs[40000] = 75
        p_post_probs[56000] = 50
        p_post_probs[75000] = 45

        s_post_probs = np.zeros(metadata["npts"])
        s_post_probs[10005] = 95
        s_post_probs[25005] = 85
        s_post_probs[40005] = 80
        s_post_probs[56005] = 55
        s_post_probs[75005] = 75

        applier.db_conn.get_channel_dates(date, "WY", "YNR", "", "HHZ")

        applier.save_daily_results_in_db(
            date,
            continuous_data,
            metadata,
            gaps,
            error,
            p_post_probs,
            s_post_probs=s_post_probs,
        )

        # check the metadata
        assert (
            applier.db_conn.daily_info.date == date
        ), "invalid date in DailyDetectionDBInfo"
        assert (
            applier.db_conn.daily_info.contdatainfo_id is not None
        ), "contdatainfo id not set"
        contdatainfo = session.get(
            tables.DailyContDataInfo, applier.db_conn.daily_info.contdatainfo_id
        )
        assert inspect(contdatainfo).persistent, "contdatainfo not persistent"
        assert contdatainfo is not None, "contdatainfo not set"
        assert contdatainfo.chan_pref == "HH", "invalid chan_pref"
        assert contdatainfo.date == date.date(), "contdatainfo date incorrect"
        assert contdatainfo.proc_start == datetime.strptime(
            "2011-09-10T00:00:00.00", datetimeformat
        ), "invalid proc_start"

        # check the gaps
        gaps_Z = services.get_gaps(
            session,
            applier.db_conn.channel_info.channel_ids["HHZ"],
            applier.db_conn.daily_info.contdatainfo_id,
        )
        assert len(gaps_Z) == 3, "Incorrect number of gaps on HHZ channel"

        # Check the Detection Output
        assert applier.db_conn.detout_storage_P is None, "detout_storage_P is set"
        assert applier.db_conn.detout_storage_S is None, "detout_storage_S is set"

        # check the detections
        det_fk_ids = applier.db_conn.get_dldet_fk_ids(is_p=True)
        inserted_dets_P = services.get_dldetections(
            session, det_fk_ids["data"], det_fk_ids["method"], 0.0, phase="P"
        )
        assert len(inserted_dets_P) == 4, "incorrect number of P detections inserted"

        # check the picks
        p_picks = services.get_picks(
            session, applier.db_conn.station_id, "HH", phase="P"
        )
        assert len(p_picks) == 3, "incorrect number of P picks"

        # check the waveforms
        for pick in p_picks:
            wf = services.get_waveforms(session, pick.id)
            assert len(wf) == 3, "invalid wf size"

        ## S
        # check the detections
        det_fk_ids = applier.db_conn.get_dldet_fk_ids(is_p=False)
        inserted_dets_S = services.get_dldetections(
            session, det_fk_ids["data"], det_fk_ids["method"], 0.0, phase="S"
        )
        assert len(inserted_dets_S) == 5, "incorrect number of S detections inserted"

        # check the picks
        s_picks = services.get_picks(
            session, applier.db_conn.station_id, "HH", phase="S"
        )
        assert len(s_picks) == 4, "incorrect number of picks"

        # check the waveforms
        for pick in s_picks:
            wf = services.get_waveforms(session, pick.id)
            assert len(wf) == 3, "invalid wf size"

    def test_apply_to_multiple_days_dumb(self, db_session, mock_pytables_config):
        session, _ = db_session
        applier = ApplyDetector(
            1, apply_detector_config, session_factory=lambda: session
        )
        applier.apply_to_multiple_days(
            "WY", "YWB", "", "EHZ", 2002, 1, 1, 2, debug_N_examples=256
        )

    def test_save_daily_results_in_db_1C_gaps_empty(
        self, db_session, contdatainfo_ex, mock_pytables_config
    ):
        session, _ = db_session
        applier = ApplyDetector(
            1, apply_detector_config, session_factory=lambda: session
        )

        date, metadata = contdatainfo_ex
        gaps = []
        error = None

        continuous_data = np.zeros((metadata["npts"], 1))
        p_post_probs = np.zeros(metadata["npts"])
        p_post_probs[10000] = 90
        p_post_probs[25000] = 80
        p_post_probs[40000] = 75
        p_post_probs[56000] = 50
        p_post_probs[75000] = 45

        applier.db_conn.get_channel_dates(date, "WY", "YNR", "", "HHZ")

        applier.save_daily_results_in_db(
            date, continuous_data, metadata, gaps, error, p_post_probs
        )

        # check the gaps
        gaps_Z = services.get_gaps(
            session,
            applier.db_conn.channel_info.channel_ids["HHZ"],
            applier.db_conn.daily_info.contdatainfo_id,
        )
        assert len(gaps_Z) == 0, "Incorrect number of gaps on HHZ channel"

        assert (
            applier.db_conn.daily_info.dldet_output_id_P is None
        ), "the detector output should not be set"

    def test_save_daily_results_in_db_1C_error(
        self, db_session, contdatainfo_ex, mock_pytables_config
    ):
        session, _ = db_session
        applier = ApplyDetector(
            1, apply_detector_config, session_factory=lambda: session
        )

        date, _ = contdatainfo_ex
        gaps = None
        error = "no_data"

        continuous_data = None
        p_post_probs = None

        applier.db_conn.get_channel_dates(date, "WY", "YNR", "", "HHZ")

        applier.save_daily_results_in_db(
            date, continuous_data, None, gaps, error, p_post_probs
        )

        # check the gaps
        gaps_Z = services.get_gaps(
            session,
            applier.db_conn.channel_info.channel_ids["HHZ"],
            applier.db_conn.daily_info.contdatainfo_id,
        )
        assert len(gaps_Z) == 0, "Incorrect number of gaps on HHZ channel"

        # check the metadata
        assert (
            applier.db_conn.daily_info.date == date
        ), "invalid date in DailyDetectionDBInfo"
        assert (
            applier.db_conn.daily_info.contdatainfo_id is not None
        ), "contdatainfo id not set"
        contdatainfo = session.get(
            tables.DailyContDataInfo, applier.db_conn.daily_info.contdatainfo_id
        )
        assert inspect(contdatainfo).persistent, "contdatainfo not persistent"
        assert contdatainfo is not None, "contdatainfo not set"
        assert contdatainfo.chan_pref == "HHZ", "invalid chan_pref"
        assert contdatainfo.date == date.date(), "contdatainfo date incorrect"
        assert contdatainfo.orig_start is None, "invalid orig_start"
        assert contdatainfo.proc_start is None, "invalid proc_start"
        assert contdatainfo.error == "no_data", "invalid error"

        assert (
            applier.db_conn.detout_storage_P is None
        ), "detout storage should not be opened"
        assert (
            applier.db_conn.daily_info.dldet_output_id_P is None
        ), "the detector output should not be set"


apply_detector_config_pytables = deepcopy(apply_detector_config)
apply_detector_config_pytables["database"]["use_pytables"] = True


class TestApplyDetectorDBPytables:
    def test_init_3c(self, db_session):
        session, _ = db_session
        applier = ApplyDetector(
            3, apply_detector_config_pytables, session_factory=lambda: session
        )
        assert applier.use_pytables, "use_pytables not set"

    def test_init_1c(self, db_session):
        session, _ = db_session
        applier = ApplyDetector(
            1, apply_detector_config_pytables, session_factory=lambda: session
        )
        assert applier.use_pytables, "use_pytables not set"

    def test_save_daily_results_in_db_1C(
        self, db_session, contdatainfo_ex, simple_obspy_gaps_ex, mock_pytables_config
    ):
        session, _ = db_session
        applier = ApplyDetector(
            1, apply_detector_config_pytables, session_factory=lambda: session
        )

        date, metadata = contdatainfo_ex
        gaps = simple_obspy_gaps_ex
        error = None

        continuous_data = np.zeros((metadata["npts"], 1))
        p_post_probs = np.zeros(metadata["npts"])
        p_post_probs[10000] = 90
        p_post_probs[25000] = 85
        p_post_probs[40000] = 80
        p_post_probs[56000] = 56
        p_post_probs[75000] = 45

        applier.db_conn.get_channel_dates(date, "WY", "YNR", "", "HHZ")
        try:
            applier.save_daily_results_in_db(
                date, continuous_data, metadata, gaps, error, p_post_probs
            )

            # check the metadata
            assert (
                applier.db_conn.daily_info.date == date
            ), "invalid date in DailyDetectionDBInfo"
            assert (
                applier.db_conn.daily_info.contdatainfo_id is not None
            ), "contdatainfo id not set"
            contdatainfo = session.get(
                tables.DailyContDataInfo, applier.db_conn.daily_info.contdatainfo_id
            )
            assert inspect(contdatainfo).persistent, "contdatainfo not persistent"
            assert contdatainfo is not None, "contdatainfo not set"
            assert contdatainfo.chan_pref == "HHZ", "invalid chan_pref"
            assert contdatainfo.date == date.date(), "contdatainfo date incorrect"
            assert contdatainfo.proc_start == datetime.strptime(
                "2011-09-10T00:00:00.00", datetimeformat
            ), "invalid proc_start"

            # check the gaps
            gaps_Z = services.get_gaps(
                session,
                applier.db_conn.channel_info.channel_ids["HHZ"],
                applier.db_conn.daily_info.contdatainfo_id,
            )
            assert len(gaps_Z) == 3, "Incorrect number of gaps on HHZ channel"

            # Check the Detection Output
            assert (
                applier.db_conn.detout_storage_P is not None
            ), "detout_storage_P not set"
            assert (
                applier.db_conn.detout_storage_P.table.nrows == 1
            ), "detout_storage should have 1 entry"
            assert (
                applier.db_conn.daily_info.dldet_output_id_P is not None
            ), "the detector output does not have an id in the db"

            # check the detections
            det_fk_ids = applier.db_conn.get_dldet_fk_ids(is_p=True)
            inserted_dets = services.get_dldetections(
                session, det_fk_ids["data"], det_fk_ids["method"], 0.0, phase="P"
            )
            assert len(inserted_dets) == 4, "incorrect number of detections inserted"

            # check the picks
            picks = services.get_picks(
                session, applier.db_conn.station_id, "HHZ", phase="P"
            )
            assert len(picks) == 3, "incorrect number of picks"

            for cid, cstore in applier.db_conn.waveform_storage_dict_P.items():
                assert cstore.table.nrows == 3, "incorrect number of waveforms saved"

            # check the waveforms
            for pick in picks:
                wf_info = services.get_waveform_infos(session, pick.id)
                assert len(wf_info) == 1, "invalid wf_info size"
                assert (
                    applier.db_conn.waveform_storage_dict_P[
                        wf_info[0].chan_id
                    ].select_row(wf_info[0].id)
                    is not None
                ), "no waveform data found for corresponding wf_info.id"
        finally:
            applier.db_conn.close_open_pytables()
            if applier.db_conn.detout_storage_P is not None:
                os.remove(applier.db_conn.detout_storage_P.file_path)
            if applier.db_conn.waveform_storage_dict_P is not None:
                for cid, cstore in applier.db_conn.waveform_storage_dict_P.items():
                    os.remove(cstore.file_path)

    def test_save_daily_results_in_db_3C(
        self, db_session, contdatainfo_ex, simple_obspy_gaps_ex, mock_pytables_config
    ):
        session, _ = db_session
        applier = ApplyDetector(
            3, apply_detector_config_pytables, session_factory=lambda: session
        )

        date, metadata = contdatainfo_ex
        gaps = simple_obspy_gaps_ex
        error = None

        continuous_data = np.zeros((metadata["npts"], 3))
        p_post_probs = np.zeros(metadata["npts"])
        p_post_probs[10000] = 90
        p_post_probs[25000] = 80
        p_post_probs[40000] = 75
        p_post_probs[56000] = 50
        p_post_probs[75000] = 45

        s_post_probs = np.zeros(metadata["npts"])
        s_post_probs[10000] = 95
        s_post_probs[25000] = 85
        s_post_probs[40000] = 80
        s_post_probs[56000] = 55
        s_post_probs[75000] = 75

        applier.db_conn.get_channel_dates(date, "WY", "YNR", "", "HH")
        try:
            applier.save_daily_results_in_db(
                date, continuous_data, metadata, gaps, error, p_post_probs, s_post_probs
            )

            # check the metadata
            assert (
                applier.db_conn.daily_info.date == date
            ), "invalid date in DailyDetectionDBInfo"
            assert (
                applier.db_conn.daily_info.contdatainfo_id is not None
            ), "contdatainfo id not set"
            contdatainfo = session.get(
                tables.DailyContDataInfo, applier.db_conn.daily_info.contdatainfo_id
            )
            assert inspect(contdatainfo).persistent, "contdatainfo not persistent"
            assert contdatainfo is not None, "contdatainfo not set"
            assert contdatainfo.chan_pref == "HH", "invalid chan_pref"
            assert contdatainfo.date == date.date(), "contdatainfo date incorrect"
            assert contdatainfo.proc_start == datetime.strptime(
                "2011-09-10T00:00:00.00", datetimeformat
            ), "invalid proc_start"

            # check the gaps
            gaps_Z = services.get_gaps(
                session,
                applier.db_conn.channel_info.channel_ids["HHZ"],
                applier.db_conn.daily_info.contdatainfo_id,
            )
            assert len(gaps_Z) == 3, "Incorrect number of gaps on HHZ channel"

            # Check the Detection Output
            assert (
                applier.db_conn.detout_storage_P is not None
            ), "detout_storage_P not set"
            assert (
                applier.db_conn.detout_storage_P.table.nrows == 1
            ), "detout_storage_P should have 1 entry"
            assert (
                applier.db_conn.daily_info.dldet_output_id_P is not None
            ), "the P detector output does not have an id in the db"

            # Check the Detection Output
            assert (
                applier.db_conn.detout_storage_S is not None
            ), "detout_storage_S not set"
            assert (
                applier.db_conn.detout_storage_S.table.nrows == 1
            ), "detout_storage_S should have 1 entry"
            assert (
                applier.db_conn.daily_info.dldet_output_id_S is not None
            ), "the S detector output does not have an id in the db"

            # check the detections
            det_fk_ids = applier.db_conn.get_dldet_fk_ids(is_p=True)
            inserted_dets = services.get_dldetections(
                session, det_fk_ids["data"], det_fk_ids["method"], 0.0, phase="P"
            )
            assert len(inserted_dets) == 4, "incorrect number of detections inserted"

            # check the picks
            p_picks = services.get_picks(
                session, applier.db_conn.station_id, "HH", phase="P"
            )
            assert len(p_picks) == 3, "incorrect number of P picks"

            for cid, cstore in applier.db_conn.waveform_storage_dict_P.items():
                assert cstore.table.nrows == 3, "incorrect number of P waveforms saved"

            # check the waveforms
            for pick in p_picks:
                wf_info = services.get_waveform_infos(session, pick.id)
                assert len(wf_info) == 3, "invalid P wf_info size"
                assert (
                    applier.db_conn.waveform_storage_dict_P[
                        wf_info[0].chan_id
                    ].select_row(wf_info[0].id)
                    is not None
                ), "no P waveform data found for corresponding wf_info.id"

            # S
            # check the detections
            det_fk_ids = applier.db_conn.get_dldet_fk_ids(is_p=False)
            s_inserted_dets = services.get_dldetections(
                session, det_fk_ids["data"], det_fk_ids["method"], 0.0, phase="S"
            )
            assert (
                len(s_inserted_dets) == 5
            ), "incorrect number of S detections inserted"

            # check the picks
            s_picks = services.get_picks(
                session, applier.db_conn.station_id, "HH", phase="S"
            )
            assert len(s_picks) == 4, "incorrect number of S picks"

            for cid, cstore in applier.db_conn.waveform_storage_dict_S.items():
                assert cstore.table.nrows == 4, "incorrect number of S waveforms saved"

            # check the waveforms
            for pick in s_picks:
                wf_info = services.get_waveform_infos(session, pick.id)
                assert len(wf_info) == 3, "invalid S wf_info size"
                assert (
                    applier.db_conn.waveform_storage_dict_S[
                        wf_info[0].chan_id
                    ].select_row(wf_info[0].id)
                    is not None
                ), "no S waveform data found for corresponding wf_info.id"

        finally:
            applier.db_conn.close_open_pytables()
            os.remove(applier.db_conn.detout_storage_P.file_path)
            os.remove(applier.db_conn.detout_storage_S.file_path)
            for cid, cstore in applier.db_conn.waveform_storage_dict_P.items():
                os.remove(cstore.file_path)
            for cid, cstore in applier.db_conn.waveform_storage_dict_S.items():
                os.remove(cstore.file_path)

    def test_apply_to_multiple_days_dumb(self, db_session, mock_pytables_config):
        session, _ = db_session
        applier = ApplyDetector(
            1, apply_detector_config_pytables, session_factory=lambda: session
        )
        try:
            applier.apply_to_multiple_days(
                "WY", "YWB", "", "EHZ", 2002, 1, 1, 2, debug_N_examples=256
            )

            # Check the Detection Output
            assert (
                not applier.db_conn.detout_storage_P._is_open
            ), "storage should have been closed within apply_to_multiple_days"
            assert (
                applier.db_conn.daily_info.dldet_output_id_P is not None
            ), "the detector output does not have an id in the db"

            for cid, cstore in applier.db_conn.waveform_storage_dict_P.items():
                assert not cstore._is_open, "waveform storage should have been closed"
        finally:
            applier.db_conn.close_open_pytables()
            os.remove(applier.db_conn.detout_storage_P.file_path)
            for cid, cstore in applier.db_conn.waveform_storage_dict_P.items():
                os.remove(cstore.file_path)

    def test_save_daily_results_in_db_1C_gaps_empty(
        self, db_session, contdatainfo_ex, mock_pytables_config
    ):
        session, _ = db_session
        applier = ApplyDetector(
            1, apply_detector_config_pytables, session_factory=lambda: session
        )

        date, metadata = contdatainfo_ex
        gaps = []
        error = None

        continuous_data = np.zeros((metadata["npts"], 1))
        p_post_probs = np.zeros(metadata["npts"])
        p_post_probs[10000] = 90
        p_post_probs[25000] = 80
        p_post_probs[40000] = 75
        p_post_probs[56000] = 50
        p_post_probs[75000] = 45

        applier.db_conn.get_channel_dates(date, "WY", "YNR", "", "HHZ")
        try:
            applier.save_daily_results_in_db(
                date, continuous_data, metadata, gaps, error, p_post_probs
            )

            # check the gaps
            gaps_Z = services.get_gaps(
                session,
                applier.db_conn.channel_info.channel_ids["HHZ"],
                applier.db_conn.daily_info.contdatainfo_id,
            )
            assert len(gaps_Z) == 0, "Incorrect number of gaps on HHZ channel"

            assert (
                applier.db_conn.daily_info.dldet_output_id_P is not None
            ), "the detector output should be set"
            assert (
                applier.db_conn.detout_storage_P.table.nrows == 1
            ), "detout storage should have 1 entry"
        finally:
            applier.db_conn.close_open_pytables()
            os.remove(applier.db_conn.detout_storage_P.file_path)
            for cid, cstore in applier.db_conn.waveform_storage_dict_P.items():
                os.remove(cstore.file_path)

    def test_save_daily_results_in_db_1C_error(
        self, db_session, contdatainfo_ex, mock_pytables_config
    ):
        session, _ = db_session
        applier = ApplyDetector(
            1, apply_detector_config_pytables, session_factory=lambda: session
        )

        date, _ = contdatainfo_ex
        gaps = None
        error = "no_data"

        continuous_data = None
        p_post_probs = None

        applier.db_conn.get_channel_dates(date, "WY", "YNR", "", "HHZ")

        applier.save_daily_results_in_db(
            date, continuous_data, None, gaps, error, p_post_probs
        )

        # check the gaps
        gaps_Z = services.get_gaps(
            session,
            applier.db_conn.channel_info.channel_ids["HHZ"],
            applier.db_conn.daily_info.contdatainfo_id,
        )
        assert len(gaps_Z) == 0, "Incorrect number of gaps on HHZ channel"

        # check the metadata
        assert (
            applier.db_conn.daily_info.date == date
        ), "invalid date in DailyDetectionDBInfo"
        assert (
            applier.db_conn.daily_info.contdatainfo_id is not None
        ), "contdatainfo id not set"
        contdatainfo = session.get(
            tables.DailyContDataInfo, applier.db_conn.daily_info.contdatainfo_id
        )
        assert inspect(contdatainfo).persistent, "contdatainfo not persistent"
        assert contdatainfo is not None, "contdatainfo not set"
        assert contdatainfo.chan_pref == "HHZ", "invalid chan_pref"
        assert contdatainfo.date == date.date(), "contdatainfo date incorrect"
        assert contdatainfo.orig_start is None, "invalid orig_start"
        assert contdatainfo.proc_start is None, "invalid proc_start"
        assert contdatainfo.error == "no_data", "invalid error"

        assert (
            applier.db_conn.detout_storage_P is None
        ), "detout storage should not be opened"
        assert (
            applier.db_conn.daily_info.dldet_output_id_P is None
        ), "the detector output should not be set"
        assert (
            applier.db_conn.waveform_storage_dict_P is None
        ), "waveform storage should not be opened"
