import pytest
from sqlalchemy.orm import sessionmaker
from sqlalchemy import engine
from datetime import datetime
from seis_proc_dl.apply_to_continuous.database_connector import DetectorDBConnection
from seis_proc_db.database import engine


dateformat = "%Y-%m-%dT%H:%M:%S.%f"

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


def test_get_channels(db_session):
    session, patch_session = db_session  # Unpack session & patch function
    db_conn = DetectorDBConnection()  # Create instance

    patch_session(db_conn)  # Patch `self.Session` on the instance
    start, end = db_conn.get_channel_dates(
        datetime.strptime("2012-10-10T00:00:00.00", dateformat), "YNR", "HH"
    )

    assert start == datetime.strptime(
        "2003-09-09T00:00:00.00", dateformat
    ), "invalid start"
    assert end == None, "invalid end"
    assert len(db_conn.channels) == 3, "invalid number of channels"
    print("CHANNEL", db_conn.channels[0])
    assert db_conn.channels[0].ondate == datetime.strptime(
        "2011-09-11T00:00:00.00", dateformat
    ), "invalid channel ondate"
    assert db_conn.channels[0].offdate == datetime.strptime(
        "2013-03-31T23:59:59.00", dateformat
    ), "invalid channel offdate"


def test_get_channels_1C(db_session):
    session, patch_session = db_session  # Unpack session & patch function
    db_conn = DetectorDBConnection()  # Create instance

    patch_session(db_conn)  # Patch `self.Session` on the instance
    start, end = db_conn.get_channel_dates(
        datetime.strptime("2012-10-10T00:00:00.00", dateformat), "QLMT", "EHZ"
    )

    assert start == datetime.strptime(
        "2001-06-09T00:00:00.00", dateformat
    ), "invalid start"
    assert end == None, "invalid end"
    assert len(db_conn.channels) == 1, "invalid number of channels"
    print("CHANNEL", db_conn.channels[0])
    assert db_conn.channels[0].ondate == datetime.strptime(
        "2003-06-10T18:00:00.00", dateformat
    ), "invalid channel ondate"
    assert db_conn.channels[0].offdate == datetime.strptime(
        "2013-09-06T18:00:00.00", dateformat
    ), "invalid channel offdate"


def test_add_detection_method_P(db_session):
    session, patch_session = db_session  # Unpack session & patch function
    db_conn = DetectorDBConnection()  # Create instance

    patch_session(db_conn)  # Patch `self.Session` on the instance

    db_conn.add_detection_method("TEST", "test method", "data/path", "P")

    assert db_conn.p_detection_method.name == "TEST", "invalid name"
    assert db_conn.p_detection_method.phase == "P", "invalid phase"


def test_add_detection_method_S(db_session):
    session, patch_session = db_session  # Unpack session & patch function
    db_conn = DetectorDBConnection()  # Create instance

    patch_session(db_conn)  # Patch `self.Session` on the instance

    db_conn.add_detection_method("TEST", "test method", "data/path", "S")

    assert db_conn.s_detection_method.name == "TEST", "invalid name"
    assert db_conn.s_detection_method.phase == "S", "invalid phase"
