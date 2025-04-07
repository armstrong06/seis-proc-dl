import sys
import obspy
import numpy as np
from seis_proc_db import database
from seis_proc_db import services

# TODO: Think I need to not store ORM objects, use them in the same session only. Just store id's and get other info as needed.


class DailyDetectionDBInfo:
    def __init__(self, date):
        self.date = date
        self.contdatainfo_id = None
        # TODO If I use bulk inserts, there won't be any detections/picks to store from the insert
        self.detections = None
        self.picks = None


class ChannelInfo:
    def __init__(self, channels):
        self.ondate = np.min([chan.ondate for chan in channels])
        ends = [chan.offdate for chan in channels]
        self.offdate = None if None in ends else np.max(ends)
        self.channel_ids = self.gather_chan_ids(channels)

    def gather_chan_ids(self, channels):
        channel_ids = {}
        for chan in channels:
            channel_ids[chan.seed_code] = chan.id
        return channel_ids


class DetectorDBConnection:
    def __init__(self, ncomps, session_factory=None):
        self.Session = session_factory or database.Session
        self.ncomps = ncomps
        self.station = None
        self.channel_info = None
        self.p_detection_method_id = None
        self.s_detection_method_id = None
        self.seed_code = None
        self.daily_info = None

    def get_channel_dates(self, date, stat, seed_code):
        """Returns the start and end times of the relevant channels for a station"""
        # The database will handel the "?" differently
        if len(seed_code) == 3 and self.ncomps == 3:
            seed_code = seed_code[:-1]
        self.seed_code = seed_code

        session = self.Session()
        with session.begin():

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
        session = self.Session()
        with session.begin():
            services.upsert_detection_method(
                session, name, phase=phase, desc=desc, path=path
            )
            if phase == "P":
                self.p_detection_method_id = services.get_detection_method(
                    session, name
                ).id
            elif phase == "S":
                self.s_detection_method_id = services.get_detection_method(
                    session, name
                ).id
            else:
                raise ValueError("Invalid Phase for Detection Method")

    def start_new_day(self, date):
        self.daily_info = DailyDetectionDBInfo(date)
        valid_channels = self.validate_channels_for_date(date)
        if not valid_channels:
            self.update_channels(date)

    def validate_channels_for_date(self, date):
        if date >= self.channel_info.ondate and (
            self.channel_info.offdate is None or date <= self.channel_info.offdate
        ):
            return True

        return False

    def update_channels(self, date):
        session = self.Session()
        with session.begin():
            # Get the Station object and the Channel objects for the appropriate date
            _, selected_channels = services.get_operating_channels_by_station_name(
                session, self.station.sta, self.seed_code, date
            )

        self.channel_info = ChannelInfo(selected_channels)

    def save_data_info(self, date, metadata_dict, error=None):
        """Add cont data info into the database. If it already exists, checks that the
        information is the same"""
        # TODO: Maybe I should add processing method table and make it part of the PK?
        # For now, I am just going to assume there is one processing method and the
        # results will be the same every time. Hopefully that's fine...

        self.start_new_day(date)

        session = self.Session()
        with session.begin():
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
                    "proc_npts": metadata_dict["npts"],
                    "proc_start": metadata_dict["starttime"],
                    "proc_end": None,  # just get this from proc_start, samp_rate, and proc_npts
                    "prev_appended": metadata_dict["previous_appended"],
                }

            contdatainfo = services.get_contdatainfo(
                session, self.station.id, self.seed_code, self.ncomps, date
            )

            if contdatainfo is None:
                contdatainfo = services.insert_contdatainfo(session, db_dict)
            else:
                # I think I did this because I'm fine with the entry existing already,
                # as long as all the values are the same. just calling insert and catching
                # the IntegrityError when there is a duplicate entry would not check
                if (
                    contdatainfo.samp_rate != db_dict["samp_rate"]
                    or contdatainfo.dt != db_dict["dt"]
                    or contdatainfo.org_npts != db_dict["org_npts"]
                    or contdatainfo.org_start != db_dict["org_start"]
                    or contdatainfo.proc_start != db_dict["proc_start"]
                    or contdatainfo.proc_npts != db_dict["proc_npts"]
                    or contdatainfo.prev_appended != db_dict["prev_appended"]
                ):
                    info_str = f"{db_dict['sta_id']}, {db_dict['chan_pref']}, {db_dict['date']}"
                    raise ValueError(
                        f"DailyContDataInfo {info_str} row already exists but the values have changed"
                    )

        self.daily_info.contdatainfo_id = contdatainfo.id

    def format_and_save_gaps(self, gaps, min_gap_sep):
        """Add gaps into the table. If many gaps in the same time period, combine them
        into one 'effective' gap. gaps should be a list of lists containing the gap seed_code, startime and endtime
        """

        assert len(gaps[0]) == 3, "Expected just three values in the gap"

        formatted_gaps = []
        for seed_code in self.channel_info.channel_ids.keys():
            chan_id = self.channel_info.channel_ids[seed_code]
            # Get only the gaps for one channel
            chan_gaps = list(filter(lambda x: x[0] == seed_code, gaps))
            # chan_inds = np.where(gap_chans == seed_code)[0]
            # chan_starts = gap_starts[chan_inds]
            # chan_ends = gap_ends[chan_inds]
            # Format the gaps for inserting
            chan_gaps = self.format_channel_gaps(chan_gaps, chan_id, min_gap_sep)
            formatted_gaps += chan_gaps

        session = self.Session()
        with session.begin():
            services.insert_gaps(session, formatted_gaps)

    def format_channel_gaps(self, gaps, chan_id, min_gap_sep):
        gaps.sort(key=lambda x: x[1])
        data_id = self.daily_info.contdatainfo_id
        merged = []
        for current in gaps:
            if not merged:
                merged.append(self.convert_gap_to_dict(current, data_id, chan_id))
            else:
                previous = merged[-1]
                # Compare the start time of the current gap to the end time of the last one
                gap_delta = (current[1] - previous["end"]).seconds
                assert gap_delta > 0, ValueError("Two adjacent gaps are overlapping")
                if gap_delta < min_gap_sep:
                    previous["end"] = current[2]
                    previous["avail_sig_sec"] += gap_delta
                else:
                    merged.append(self.convert_gap_to_dict(current, data_id, chan_id))

        return merged

    @staticmethod
    def convert_gap_to_dict(gap, data_id, chan_id):
        d = {
            "data_id": data_id,
            "chan_id": chan_id,
            "start": gap[1],
            "end": gap[2],
            "avail_sig_sec": 0.0,
        }
        return d

    def save_detections(self, detections):
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
