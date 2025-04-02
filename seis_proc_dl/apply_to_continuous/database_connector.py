import sys
import obspy
import numpy as np

from seis_proc_db import database
from seis_proc_db import services


class DetectorDBConnection:
    def __init__(self):
        self.station = None
        self.channels = None
        self.p_detection_method = None
        self.s_detection_method = None
        self.Session = database.Session

    def get_channel_dates(self, date, stat, seed_code):
        """Returns the start and end times of the relevant channels for a station"""
        # The database will handel the "?" differently
        if len(seed_code) == 3 and seed_code[-1] == "?":
            seed_code = seed_code[:-1]

        with self.Session() as session:

            # Get all channels for this station name and channel type
            all_channels = services.get_common_station_channels_by_name(
                session, stat, seed_code
            )

            # Get the Station object and the Channel objects for the appropriate date
            selected_stat, selected_channels = (
                services.get_operating_channels_by_station_name(
                    session, stat, seed_code, date
                )
            )

        if selected_stat is None:
            return None, None

        # Store the station
        self.station = selected_stat
        # Store the current channels
        self.channels = selected_channels

        # Set the start date to the minimum date for all channels of the desired type
        start_date = np.min([chan.ondate for chan in all_channels])
        # Set the end date to the maximum date for all channels of the desired type
        ends = [chan.offdate for chan in all_channels]
        end_date = None if None in ends else np.max(ends)

        return start_date, end_date

    def add_detection_method(self, name, desc, path, phase):
        """Add a detection method to the database. If it already exists, update it."""
        with self.Session() as session:
            services.upsert_detection_method(
                session, name, phase=phase, desc=desc, path=path
            )
            if phase == "P":
                self.p_detection_method = services.get_detection_method(session, name)
            elif phase == "S":
                self.s_detection_method = services.get_detection_method(session, name)
            else:
                raise ValueError("Invalid Phase for Detection Method")

    def save_data_info():
        """Add cont data info into the database. If it already exists, checks that the
        information is the same"""
        # TODO: Maybe I should add processing method table and make it part of the PK?
        pass

    def save_gaps():
        """Add gaps into the table. If many gaps in the same time period, combine them
        into one 'effective' gap"""
        pass

    def save_detections():
        """Add detections above a threshold into the database. Do not add detections if
        they exist within a gap"""
        pass

    def make_picks_from_detections():
        """Add detections above a certain threshold into the pick table, with the
        necessary additional information"""
        pass

    def save_waveforms():
        """Extract waveforms around a pick from all channels and store in the database"""
        pass
