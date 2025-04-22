import sys
import obspy
import numpy as np
from copy import deepcopy
from datetime import timedelta
from seis_proc_db import database
from seis_proc_db import services
from seis_proc_db import pytables_backend
from seis_proc_db.tables import Waveform, DailyContDataInfo, Channel, DLDetection

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

        self.station_name = None
        self.seed_code = None

        self.station_id = None
        self.channel_info = None
        self.p_detection_method_id = None
        self.s_detection_method_id = None
        self.daily_info = None

        # BasePyTables storage
        # Wavefroms will need a storage for each channel - keep them in dicts
        self.waveform_storage_dict_P = None
        self.waveform_storage_dict_S = None
        # Only need one storage per phase for detection outputs
        self.detout_storage_P = None
        self.detout_storage_S = None

    def get_channel_dates(self, date, stat, seed_code):
        """Returns the start and end times of the relevant channels for a station"""
        # The database will handel the "?" differently
        if len(seed_code) == 3 and self.ncomps == 3:
            seed_code = seed_code[:-1]
        self.seed_code = seed_code
        self.station_name = stat
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

        # Set the start date to the minimum date for all channels of the desired type
        start_date = np.min([chan.ondate for chan in all_channels])
        # Set the end date to the maximum date for all channels of the desired type
        ends = [chan.offdate for chan in all_channels]
        end_date = None if None in ends else np.max(ends)

        if selected_stat is not None:
            # If there is a selected stat,
            assert (
                len(selected_channels) == self.ncomps
            ), f"Number of channels selected ({len(selected_channels)}) does not agree with the number of components ({self.ncomps})"
            # Store the station
            self.station_id = selected_stat.id
            # Store the current channels
            self.channel_info = ChannelInfo(selected_channels)

        return start_date, end_date

    def add_detection_method(self, name, desc, path, phase):
        """Add a detection method to the database. If it already exists, update it."""
        session = self.Session()
        with session.begin():
            services.upsert_detection_method(
                session, name, phase=phase, details=desc, path=path
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
                session, self.station_name, self.seed_code, date
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
                "sta_id": self.station_id,
                "chan_pref": self.seed_code,
                "ncomps": self.ncomps,
                "date": date,
                "error": error,
            }

            if metadata_dict is not None:
                db_dict = db_dict | {
                    "samp_rate": metadata_dict["sampling_rate"],
                    "dt": metadata_dict["dt"],
                    "orig_npts": metadata_dict["original_npts"],
                    "orig_start": metadata_dict["original_starttime"],
                    "orig_end": metadata_dict["original_endtime"],
                    "proc_npts": metadata_dict["npts"] if error is None else None,
                    "proc_start": metadata_dict["starttime"] if error is None else None,
                    "proc_end": None,  # just get this from proc_start, samp_rate, and proc_npts
                    "prev_appended": metadata_dict["previous_appended"],
                }

            contdatainfo = services.get_contdatainfo(
                session, self.station_id, self.seed_code, self.ncomps, date
            )

            if contdatainfo is None:
                contdatainfo = services.insert_contdatainfo(session, db_dict)
            elif metadata_dict is None:
                if contdatainfo.error != db_dict["error"]:
                    info_str = f"{db_dict['sta_id']}, {db_dict['chan_pref']}, {db_dict['date']}"
                    raise ValueError(
                        f"DailyContDataInfo {info_str} row already exists but the error when loading has changed"
                    )
            else:
                # I think I did this because I'm fine with the entry existing already,
                # as long as all the values are the same. just calling insert and catching
                # the IntegrityError when there is a duplicate entry would not check
                if (
                    contdatainfo.samp_rate != db_dict["samp_rate"]
                    or contdatainfo.dt != db_dict["dt"]
                    or contdatainfo.orig_npts != db_dict["orig_npts"]
                    or contdatainfo.orig_start != db_dict["orig_start"]
                    or contdatainfo.proc_start != db_dict["proc_start"]
                    or contdatainfo.proc_npts != db_dict["proc_npts"]
                    or contdatainfo.prev_appended != db_dict["prev_appended"]
                    or contdatainfo.error != db_dict["error"]
                ):
                    info_str = f"{db_dict['sta_id']}, {db_dict['chan_pref']}, {db_dict['date']}"
                    raise ValueError(
                        f"DailyContDataInfo {info_str} row already exists but the values have changed"
                    )

        self.daily_info.contdatainfo_id = contdatainfo.id

    def format_and_save_gaps(self, gaps, min_gap_sep_seconds):
        """Add gaps into the table. If many gaps in the same time period, combine them
        into one 'effective' gap. gaps should be a list of lists containing the gap seed_code, startime and endtime
        """

        if gaps is None or len(gaps) == 0:
            return

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
            chan_gaps = self.format_channel_gaps(
                chan_gaps, chan_id, min_gap_sep_seconds
            )
            formatted_gaps += chan_gaps

        session = self.Session()
        with session.begin():
            services.insert_gaps(session, formatted_gaps)

    def format_channel_gaps(self, gaps, chan_id, min_gap_sep_seconds):
        if gaps is None:
            return []

        gaps.sort(key=lambda x: x[1])
        data_id = self.daily_info.contdatainfo_id
        merged = []
        for current in gaps:
            if not merged:
                merged.append(self.convert_gap_to_dict(current, data_id, chan_id))
            else:
                previous = merged[-1]
                # Compare the start time of the current gap to the end time of the last one
                gap_delta = (current[1] - previous["end"]).total_seconds()
                assert gap_delta > 0, ValueError("Two adjacent gaps are overlapping")
                if gap_delta < min_gap_sep_seconds:
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
            # (gap[1] if type(gap[1]) == datetime else gap[1].datetime),
            "start": gap[1],
            # (gap[2] if type(gap[2]) == datetime else gap[2].datetime),
            "end": gap[2],
            "avail_sig_sec": 0.0,
        }
        return d

    def save_detections(self, detections):
        """Add detections above a threshold into the database. Do not add detections if
        they exist within a gap"""

        if len(detections) == 0:
            return

        session = self.Session()
        with session.begin():
            services.bulk_insert_dldetections_with_gap_check(session, detections)

    def get_dldet_fk_ids(self, is_p=True):
        d = {"data": self.daily_info.contdatainfo_id}
        if is_p:
            d["method"] = self.p_detection_method_id
        else:
            d["method"] = self.s_detection_method_id
        return d

    def save_P_post_probs(self, data, expected_array_length=8640000, on_event=None):
        if self.detout_storage_P is None:
            self.detout_storage_P = self._open_dldetection_output_storage(
                expected_array_length=expected_array_length,
                phase="P",
                det_method_id=self.p_detection_method_id,
                on_event=on_event,
            )

        self._save_detection_output(
            self.detout_storage_P, data, self.p_detection_method_id
        )

    def save_S_post_probs(self, data, expected_array_length=8640000, on_event=None):
        if self.detout_storage_S is None:
            self.detout_storage_S = self._open_dldetection_output_storage(
                expected_array_length=expected_array_length,
                phase="S",
                det_method_id=self.s_detection_method_id,
                on_event=on_event,
            )

        self._save_detection_output(
            self.detout_storage_S, data, self.s_detection_method_id
        )

    def _open_dldetection_output_storage(
        self, expected_array_length, phase, det_method_id, on_event=None
    ):
        storage = pytables_backend.DLDetectorOutputStorage(
            expected_array_length=expected_array_length,
            sta=self.station_name,
            seed_code=self.seed_code,
            ncomps=self.ncomps,
            phase=phase,
            det_method_id=det_method_id,
            on_event=on_event,
        )
        return storage

    def _save_detection_output(self, storage, data, det_method_id):
        session = self.Session()
        with session.begin():
            try:
                storage.begin_transaction()
                services.insert_dldetector_output_pytable(
                    session,
                    storage,
                    self.daily_info.contdatainfo_id,
                    det_method_id,
                    data.astype(np.uint8),
                )
            except Exception as e:
                storage.rollback()
                raise e

        storage.commit()

    def open_waveform_storages(
        self,
        expected_array_length,
        phase,
        filt_low,
        filt_high,
        proc_notes,
        channel_ids,
        on_event=None,
    ):
        pytables_storage = {}
        for seed_code, chan_id in channel_ids.items():
            pytables_storage[seed_code] = pytables_backend.WaveformStorage(
                expected_array_length=expected_array_length,
                sta=self.station_name,
                seed_code=seed_code,
                ncomps=self.ncomps,
                phase=phase,
                filt_low=filt_low,
                filt_high=filt_high,
                proc_notes=proc_notes,
                on_event=on_event,
            )

        return pytables_storage

    def save_picks_from_detections(
        self,
        pick_thresh,
        is_p,
        auth,
        continuous_data,
        wf_filt_low,
        wf_filt_high,
        wf_proc_notes,
        seconds_around_pick,
    ):
        """Add detections above a certain threshold into the pick table, with the
        necessary additional information"""

        session = self.Session()
        with session.begin():
            # Get ids
            data_id = self.daily_info.contdatainfo_id
            if is_p:
                method_id = self.p_detection_method_id
                phase = "P"
            else:
                method_id = self.s_detection_method_id
                phase = "S"

            # Comput the number of samples to grab on either side of the detection
            cdi = session.get(DailyContDataInfo, data_id)
            samples_around_pick = int(seconds_around_pick * cdi.samp_rate)
            total_npts = len(continuous_data)

            # Get all detections for the contdatainfo and method greater than the pick_thresh
            dldets = services.get_dldetections(
                session, data_id, method_id, pick_thresh, phase=phase
            )

            # Iterate over the detections
            for det in dldets:

                # Compute the relevant waveform information for all channels
                i1 = det.sample - samples_around_pick
                i2 = det.sample + samples_around_pick + 1
                if i1 < 0:
                    i1 = 0
                # TODO: Check if this needs a -1
                if i2 > total_npts:
                    i2 = total_npts
                pick_cont_data = deepcopy(continuous_data[i1:i2, :])
                wf_start = cdi.proc_start + timedelta(seconds=(i1 * cdi.dt))
                wf_end = cdi.proc_start + timedelta(seconds=(i2 * cdi.dt))

                # Check if the detection is on the previous day, if so need to check
                # for existing picks and handle accordingly
                insert_new_pick = True
                if det.time.date() < cdi.date:
                    assert (
                        cdi.prev_appended
                    ), "Previous data was not appended, yet there is a detection on the previous day..."

                    # Check if there are any picks with a ptime close to det.time
                    close_picks = services.get_picks(
                        session,
                        self.station_id,
                        self.seed_code,
                        phase,
                        min_time=det.time - timedelta(seconds=0.1),
                        max_time=det.time + timedelta(seconds=0.1),
                    )

                    # If there are no close picks, then insert_new_pick = True
                    if len(close_picks) == 0:
                        continue
                    if len(close_picks) > 1:
                        raise NotImplementedError(
                            "There are multiple close picks in the previous day's data..."
                        )

                    # If made it to this point, will update or keep an existing pick
                    insert_new_pick = False

                    # There's only one pick
                    pick = close_picks[0]
                    prev_data_id = session.get(DLDetection, pick.detid).data_id
                    # Get the waveforms for these picks
                    close_wfs = services.get_waveforms(
                        session, pick.id, data_id=prev_data_id
                    )

                    # Only update the pick and waveforms if the waveforms would have more continuous data available
                    prev_inserted_npts = len(close_wfs[0].data)
                    if prev_inserted_npts > (i2 - i1):
                        continue

                    # If so, update the pick time and detection id, everything else should be the same
                    pick.ptime = det.time
                    pick.detid = det.id

                    # Update the waveforms assigned to the pick
                    for wf in close_wfs:
                        seed_code = session.get(Channel, wf.chan_id).seed_code
                        # Get the appropriate channel index of the data
                        chan_ind = self.get_channel_data_index(self.ncomps, seed_code)
                        # Get just the channel of interest
                        wf_data = pick_cont_data[:, chan_ind].tolist()

                        wf.data = wf_data
                        wf.start = wf_start
                        wf.end = wf_end
                        wf.data_id = data_id
                        # TODO: Might want to update this in case the channel switched between days...
                        # But I think it makes sense to still assign it to the previous day's channel
                        # wf.chan_id = self.channel_info.channel_ids[seed_code]

                if insert_new_pick:
                    # Create a pick from the detection
                    pick = services.insert_pick(
                        session,
                        self.station_id,
                        self.seed_code,
                        det.phase,
                        det.time,
                        auth,
                        detid=det.id,
                    )

                    # print(wf_end - det.time)
                    # print(det.time - wf_start)
                    # expected_start = det.time - timedelta(seconds=seconds_around_pick)
                    # expected_end = det.time + timedelta(
                    #     seconds=seconds_around_pick + cdi.dt
                    # )
                    # datetimeformat = "%Y-%m-%dT%H:%M:%S.%f"
                    # print(
                    #     "START",
                    #     expected_start.strftime(datetimeformat),
                    #     wf_start.strftime(datetimeformat),
                    # )
                    # print(
                    #     "END",
                    #     expected_end.strftime(datetimeformat),
                    #     wf_end.strftime(datetimeformat),
                    # )
                    # assert (
                    #     wf_start - expected_start
                    # ).microseconds == 0, "wf_start different than expected"
                    # assert (
                    #     wf_end - expected_end
                    # ).microseconds == 0, "wf_end different than expected"

                    # Iterate over the different channels
                    for seed_code in self.channel_info.channel_ids.keys():
                        chan_id = self.channel_info.channel_ids[seed_code]

                        # Get the appropriate channel index of the data
                        chan_ind = self.get_channel_data_index(self.ncomps, seed_code)

                        # Get just the channel of interest
                        wf_data = pick_cont_data[:, chan_ind].tolist()

                        # Create the waveform object
                        wf = Waveform(
                            data_id=data_id,
                            chan_id=chan_id,
                            filt_low=wf_filt_low,
                            filt_high=wf_filt_high,
                            data=wf_data,
                            start=wf_start,
                            end=wf_end,
                            proc_notes=wf_proc_notes,
                        )
                        # Add the waveform to the pick
                        pick.wfs.add(wf)

    @staticmethod
    def get_channel_data_index(ncomps, seed_code):
        # Get the appropriate channel index of the data
        if ncomps == 1:
            chan_ind = 0
        elif seed_code in ["EHZ", "BHZ", "HHZ"]:
            chan_ind = 2
        elif seed_code in ["EHE", "EH1", "BHE", "BH1", "HHE", "HH1"]:
            chan_ind = 0
        elif seed_code in ["EHN", "EH2", "BHN", "BH2", "HHN", "HH2"]:
            chan_ind = 1
        else:
            raise ValueError("Something is wrong with the channel code")

        return chan_ind
