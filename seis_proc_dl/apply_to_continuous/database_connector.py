import sys
import obspy
import numpy as np

from seis_proc_db import database
from seis_proc_db import services

class DailyDetectionDBInfo():
    def __init__(self, date):
        self.date = date
        self.contdatainfo = None
        self.gaps = None
        self.detections = None
        self.picks = None


class ChannelInfo():
    def __init__(self, channels):
        self.channels = channels
        self.startdate = np.min([chan.ondate for chan in channels])
        ends = [chan.offdate for chan in channels]
        self.enddate = None if None in ends else np.max(ends)

class DetectorDBConnection:
    def __init__(self, ncomps):
        self.ncomps = ncomps
        self.station = None
        self.channel_info = None
        self.p_detection_method = None
        self.s_detection_method = None
        self.seed_code = None
        self.Session = database.Session
        self.daily_info = None

    def get_channel_dates(self, date, stat, seed_code):
        """Returns the start and end times of the relevant channels for a station"""
        # The database will handel the "?" differently
        if len(seed_code) == 3 and self.ncomps == 3:
            seed_code = seed_code[:-1]
        self.seed_code = seed_code

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
        self.channel_info = ChannelInfo(selected_channels)

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

    def start_new_day(self, date):
        self.daily_info = DailyDetectionDBInfo(date)
        valid_channels = self.validate_channels_for_date(date)
        if not valid_channels:
            self.update_channels(date)

    def validate_channels_for_date(self, date):
        if (date >= self.channel_info.startdate and 
            (self.channel_info.enddate is None or self.channel_info.enddate <= date)):
            return True
            
        return False
    
    def update_channels(self, date):
        with self.Session() as session:
        # Get the Station object and the Channel objects for the appropriate date
            _, selected_channels = (
                services.get_operating_channels_by_station_name(
                    session, self.station.sta, self.seed_code, date
                ))
        
        self.channel_info = ChannelInfo(selected_channels)

    def save_data_info(self, date, metadata_dict, error=None):
        """Add cont data info into the database. If it already exists, checks that the
        information is the same"""
        # TODO: Maybe I should add processing method table and make it part of the PK?
        # For now, I am just going to assume there is one processing method and the
        # results will be the same every time. Hopefully that's fine...

        self.start_new_day(date)

        with self.Session() as session:
            # assert metadata_dict["chan_pref"][0:2] == self.seed_code
            contdatainfo = services.get_contdatainfo(
                session, self.station.id, self.seed_code, self.ncomps, date
            )
            db_dict = {
                "sta_id": self.station.id,
                "chan_pref": self.seed_code,
                "ncomps": self.ncomps,
                "date": date,
                "error": error,
            }

            if metadata_dict is not None:
                db_dict = db_dict | {
                    "samp_rate": metadata_dict["sampling_rate"],
                    "dt": metadata_dict["dt"],
                    "org_npts": metadata_dict["original_npts"],
                    "org_start": metadata_dict["original_starttime"],
                    "org_end": metadata_dict["original_endtime"],
                    "proc_npts": proc_npts,
                    "proc_start": proc_start,
                    "proc_end": None,  # just get this from proc_start, samp_rate, and proc_npts
                    "prev_appended": metadata_dict["previous_appended"],
                    
                    }
             
            if contdatainfo is None:
                proc_npts = metadata_dict["npts"]
                proc_start = metadata_dict["starttime"]

                if metadata_dict["orginal_npts"] == proc_npts:
                    proc_npts = None

                if metadata_dict["original_starttime"] == proc_start:
                    proc_start = None

                contdatainfo = services.insert_contdatainfo(session, **db_dict)
            else:
                if (
                    contdatainfo.samp_rate != db_dict["samp_rate"]
                    and contdatainfo.dt != db_dict["dt"]
                    and contdatainfo.org_npts != db_dict["org_npts"]
                    and contdatainfo.org_start != db_dict["org_start"]
                    and contdatainfo.proc_npts != db_dict["proc_start"]
                    and contdatainfo.prev_appended != db_dict["prev_appended"]
                ):
                    info_str = f"{db_dict["sta_id"]}, {db_dict["chan_pref"]}, {db_dict["date"]}"
                    raise ValueError(
                        f"DailyContDataInfo {info_str} row already exists but the values have changed"
                    )

        self.daily_info.contdatainfo = contdatainfo

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
