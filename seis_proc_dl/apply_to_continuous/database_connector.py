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
        self.dldet_output_id_P = None
        self.dldet_output_id_S = None


class ChannelInfo:
    def __init__(self, channels, total_ndays):
        self.ondate = None
        self.offdate = None
        self.channel_ids = None
        self.ndays = total_ndays

        if channels is not None and len(channels) > 0:
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
    EXPECTED_DAILY_P_PICKS = 1000
    EXPECTED_DAILY_S_PICKS = 1000

    def __init__(self, ncomps, session_factory=None):
        self.Session = session_factory or database.Session
        self.ncomps = ncomps

        self.station_name = None
        self.seed_code = None
        self.net = None
        self.loc = None

        self.station_id = None
        self.channel_info = None
        self.p_detection_method_id = None
        self.s_detection_method_id = None
        self.daily_info = None
        self.wf_source_id = None

        # BasePyTables storage
        # Wavefroms will need a storage for each channel - keep them in dicts
        self.waveform_storage_dict_P = None
        self.waveform_storage_dict_S = None
        # Only need one storage per phase for detection outputs
        self.detout_storage_P = None
        self.detout_storage_S = None

    def get_channel_dates(self, date, net, stat, loc, seed_code):
        """Returns the start and end times of the relevant channels for a station"""
        # The database will handel the "?" differently
        if len(seed_code) == 3 and self.ncomps == 3:
            seed_code = seed_code[:-1]
        self.seed_code = seed_code
        self.station_name = stat
        self.net = net
        self.loc = loc
        with self.Session() as session:
            with session.begin():
                # Get all channels for this station name and channel type
                all_channels = services.get_common_station_channels_by_name(
                    session, stat, seed_code, net=net, loc=loc
                )

                # Get the Station object and the Channel objects for the appropriate date
                selected_stat, selected_channels = (
                    services.get_operating_channels_by_station_name(
                        session, stat, seed_code, date, net=net, loc=loc
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
                    total_ndays = services.get_similar_channel_total_ndays(
                        session, net, stat, loc, selected_channels[0].seed_code
                    )
                    self.channel_info = ChannelInfo(selected_channels, total_ndays)

        return start_date, end_date

    def add_waveform_source(self, name, desc, path=None):
        """Add a waveform source to the database. If it already exists, update it."""
        with self.Session() as session:
            with session.begin():
                services.upsert_waveform_source(
                    session, name, details=desc, path=path
                )
                self.wf_source_id = services.get_waveform_source(session, name).id

    def add_detection_method(self, name, desc, path, phase):
        """Add a detection method to the database. If it already exists, update it."""
        with self.Session() as session:
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
            channel_continues = self.update_channels(date)

            if not channel_continues:
                return False

        return True

    def validate_channels_for_date(self, date):
        if date >= self.channel_info.ondate and (
            self.channel_info.offdate is None or date <= self.channel_info.offdate
        ):
            return True

        return False

    def update_channels(self, date):
        with self.Session() as session:
            with session.begin():
                # Get the Station object and the Channel objects for the appropriate date
                _, selected_channels = services.get_operating_channels_by_station_name(
                    session,
                    self.station_name,
                    self.seed_code,
                    date,
                    net=self.net,
                    loc=self.loc,
                )

                if selected_channels is None or len(selected_channels) == 0:
                    self.channel_info = ChannelInfo([], 0)
                    return False

                total_ndays = services.get_similar_channel_total_ndays(
                    session,
                    self.net,
                    self.station_name,
                    self.loc,
                    selected_channels[0].seed_code,
                )

        self.channel_info = ChannelInfo(selected_channels, total_ndays)
        return True

    def save_data_info(self, date, metadata_dict, error=None):
        """Add cont data info into the database. If it already exists, checks that the
        information is the same"""
        # TODO: Maybe I should add processing method table and make it part of the PK?
        # For now, I am just going to assume there is one processing method and the
        # results will be the same every time. Hopefully that's fine...

        started_new_day = self.start_new_day(date)
        if not started_new_day:
            raise ValueError(f"Cannot start processing {date}, no channel info exists")

        with self.Session() as session:
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
                        "proc_start": (
                            metadata_dict["starttime"] if error is None else None
                        ),
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

        with self.Session() as session:
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
                assert gap_delta >= 0, ValueError("Two adjacent gaps are overlapping")
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

        with self.Session() as session:
            with session.begin():
                services.bulk_insert_dldetections_with_gap_check(session, detections)

    def get_dldet_fk_ids(self, is_p=True):
        d = {
            "data": self.daily_info.contdatainfo_id,
        }
        if is_p:
            d["method"] = self.p_detection_method_id
            d["detout"] = self.daily_info.dldet_output_id_P
        else:
            d["method"] = self.s_detection_method_id
            d["detout"] = self.daily_info.dldet_output_id_S
        return d

    def save_P_post_probs(self, data, expected_array_length=8640000, on_event=None):
        if self.detout_storage_P is None:
            self.detout_storage_P = self._open_dldetection_output_storage(
                expected_array_length=expected_array_length,
                phase="P",
                det_method_id=self.p_detection_method_id,
                on_event=on_event,
            )

        if len(data) < expected_array_length:
            tmp = np.zeros((int(expected_array_length - len(data)),), dtype=np.uint8)
            data = np.concatenate([data, tmp])

        detout_id = self._save_detection_output(
            self.detout_storage_P,
            data,
            self.p_detection_method_id,
        )
        self.daily_info.dldet_output_id_P = detout_id

    def save_S_post_probs(self, data, expected_array_length=8640000, on_event=None):
        if self.detout_storage_S is None:
            self.detout_storage_S = self._open_dldetection_output_storage(
                expected_array_length=expected_array_length,
                phase="S",
                det_method_id=self.s_detection_method_id,
                on_event=on_event,
            )

        if len(data) < expected_array_length:
            tmp = np.zeros((int(expected_array_length - len(data)),), dtype=np.uint8)
            data = np.concatenate([data, tmp])

        detout_id = self._save_detection_output(
            self.detout_storage_S, data, self.s_detection_method_id
        )
        self.daily_info.dldet_output_id_S = detout_id

    def _open_dldetection_output_storage(
        self, expected_array_length, phase, det_method_id, on_event=None
    ):
        storage = pytables_backend.DLDetectorOutputStorage(
            expected_array_length=expected_array_length,
            net=self.net,
            sta=self.station_name,
            loc=self.loc,
            seed_code=self.seed_code,
            ncomps=self.ncomps,
            phase=phase,
            det_method_id=det_method_id,
            on_event=on_event,
            expectedrows=self.channel_info.ndays,
        )
        return storage

    def _save_detection_output(self, storage, data, det_method_id):
        detout_id = None
        with self.Session() as session:
            with session.begin():
                try:
                    storage.start_transaction()
                    detout = services.insert_dldetector_output_pytable(
                        session,
                        storage,
                        self.daily_info.contdatainfo_id,
                        det_method_id,
                        data.astype(np.uint8),
                    )
                    detout_id = detout.id
                except Exception as e:
                    storage.rollback()
                    self.close_open_pytables()
                    raise e

        storage.commit()
        return detout_id

    def _get_waveform_storages(self, is_p, common_wf_details):

        if (is_p and self.waveform_storage_dict_P is None) or (
            not is_p and self.waveform_storage_dict_S is None
        ):
            pytables_storage = {}
            for seed_code, chan_id in self.channel_info.channel_ids.items():
                pytables_storage[chan_id] = pytables_backend.WaveformStorage(
                    expected_array_length=common_wf_details["expected_array_length"],
                    net=self.net,
                    sta=self.station_name,
                    loc=self.loc,
                    seed_code=seed_code,
                    ncomps=self.ncomps,
                    phase=common_wf_details["phase"],
                    filt_low=common_wf_details["wf_filt_low"],
                    filt_high=common_wf_details["wf_filt_high"],
                    proc_notes=common_wf_details["wf_proc_notes"],
                    on_event=common_wf_details["on_event"],
                    expectedrows=(
                        self.EXPECTED_DAILY_P_PICKS * self.channel_info.ndays
                        if is_p
                        else self.EXPECTED_DAILY_S_PICKS * self.channel_info.ndays
                    ),
                )

            if is_p:
                self.waveform_storage_dict_P = pytables_storage
            else:
                self.waveform_storage_dict_S = pytables_storage

        else:
            if is_p:
                pytables_storage = self.waveform_storage_dict_P
            else:
                pytables_storage = self.waveform_storage_dict_S

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
        use_pytables=True,
        on_event=None,
    ):
        """Add detections above a certain threshold into the pick table, with the
        necessary additional information"""

        common_wf_details = {
            "wf_filt_low": wf_filt_low,
            "wf_filt_high": wf_filt_high,
            "wf_proc_notes": wf_proc_notes,
            "use_pytables": use_pytables,
            "on_event": on_event,
        }
        storage_dict = None

        with self.Session() as session:
            with session.begin():
                # Get ids
                data_id = self.daily_info.contdatainfo_id
                if is_p:
                    method_id = self.p_detection_method_id
                    phase = "P"
                else:
                    method_id = self.s_detection_method_id
                    phase = "S"
                # Compute the number of samples to grab on either side of the detection
                cdi = session.get(DailyContDataInfo, data_id)
                samples_around_pick = int(seconds_around_pick * cdi.samp_rate)
                total_npts = len(continuous_data)
                total_expected_samples = samples_around_pick * 2 + 1

                common_wf_details["expected_array_length"] = total_expected_samples
                common_wf_details["phase"] = phase
                common_wf_details["data_id"] = data_id
                try:
                    #### GET THE PYTABLES STORAGE
                    storage_dict = None
                    if use_pytables:
                        storage_dict = self._get_waveform_storages(
                            is_p, common_wf_details
                        )

                        # Start a transaction for these
                        for _, chan_storage in storage_dict.items():
                            chan_storage.start_transaction()
                    #####

                    # Get all detections for the contdatainfo and method greater than the pick_thresh
                    dldets = services.get_dldetections(
                        session, data_id, method_id, pick_thresh, phase=phase
                    )

                    # Iterate over the detections
                    for det in dldets:
                        pick_waveform_details = {}

                        ## Compute the start and end inds for waveforms in pytable
                        ## Rely on the default values in the storage object
                        wf_start_ind = 0
                        wf_end_ind = total_expected_samples
                        ##

                        # Compute the relevant waveform information for all channels
                        i1 = det.sample - samples_around_pick
                        i2 = det.sample + samples_around_pick + 1
                        if i1 < 0:
                            wf_start_ind = abs(i1)
                            i1 = 0
                        # TODO: Check if this needs a -1
                        if i2 > total_npts:
                            wf_end_ind -= i2 - total_npts
                            i2 = total_npts
                        pick_cont_data = deepcopy(continuous_data[i1:i2, :])
                        wf_start = cdi.proc_start + timedelta(seconds=(i1 * cdi.dt))
                        wf_end = cdi.proc_start + timedelta(seconds=(i2 * cdi.dt))
                        pick_waveform_details["npts"] = i2 - i1
                        pick_waveform_details["wf_start"] = wf_start
                        pick_waveform_details["wf_end"] = wf_end
                        pick_waveform_details["wf_start_ind"] = wf_start_ind
                        pick_waveform_details["wf_end_ind"] = wf_end_ind
                        pick_waveform_details["pick_cont_data"] = pick_cont_data
                        # Check if the detection is on the previous day, if so need to check
                        # for existing picks and handle accordingly
                        insert_new_pick = True
                        if det.time.date() < cdi.date:
                            assert (
                                cdi.prev_appended
                            ), "Previous data was not appended, yet there is a detection on the previous day..."

                            insert_new_pick = (
                                self._potentially_modify_pick_and_waveform(
                                    session,
                                    storage_dict,
                                    det,
                                    pick_waveform_details,
                                    common_wf_details,
                                )
                            )

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
                            self._insert_new_waveforms(
                                session,
                                storage_dict,
                                pick,
                                pick_waveform_details,
                                common_wf_details,
                            )
                except Exception as e:
                    if use_pytables:
                        # Rollback the pytables updates if an error occurs
                        for _, chan_storage in storage_dict.items():
                            chan_storage.rollback()
                        self.close_open_pytables()

                    raise e

        if use_pytables:
            # Commit the pytables updates if everything went well in the database transaction
            for _, chan_storage in storage_dict.items():
                chan_storage.commit()

    def _insert_new_waveforms(
        self, session, storage_dict, pick, pick_wf_details, common_wf_details
    ):
        pick_cont_data = pick_wf_details["pick_cont_data"]
        for seed_code in self.channel_info.channel_ids.keys():
            chan_id = self.channel_info.channel_ids[seed_code]

            # Get the appropriate channel index of the data
            chan_ind = self.get_channel_data_index(self.ncomps, seed_code)

            # Get just the channel of interest
            wf_data = pick_cont_data[:, chan_ind]

            if not common_wf_details["use_pytables"]:
                # Create the waveform object
                wf = Waveform(
                    data_id=common_wf_details["data_id"],
                    chan_id=chan_id,
                    filt_low=common_wf_details["wf_filt_low"],
                    filt_high=common_wf_details["wf_filt_high"],
                    data=wf_data.tolist(),
                    start=pick_wf_details["wf_start"],
                    end=pick_wf_details["wf_end"],
                    proc_notes=common_wf_details["wf_proc_notes"],
                )
                # Add the waveform to the pick
                pick.wfs.add(wf)
            else:
                # Need to flush to get the pick id
                session.flush()
                chan_storage = storage_dict[chan_id]
                pytables_wf_data = np.zeros(
                    common_wf_details["expected_array_length"], dtype=np.float32
                )
                pytables_wf_data[
                    pick_wf_details["wf_start_ind"] : pick_wf_details["wf_end_ind"]
                ] = wf_data
                _ = services.insert_waveform_pytable(
                    session,
                    chan_storage,
                    data=pytables_wf_data,
                    data_id=common_wf_details["data_id"],
                    chan_id=chan_id,
                    pick_id=pick.id,
                    wf_source_id=self.wf_source_id,
                    start=pick_wf_details["wf_start"],
                    end=pick_wf_details["wf_end"],
                    filt_low=common_wf_details["wf_filt_low"],
                    filt_high=common_wf_details["wf_filt_high"],
                    proc_notes=common_wf_details["wf_proc_notes"],
                    signal_start_ind=pick_wf_details["wf_start_ind"],
                    signal_end_ind=pick_wf_details["wf_end_ind"],
                )

    def _potentially_modify_pick_and_waveform(
        self,
        session,
        storage_dict,
        detection,
        pick_waveform_details,
        common_waveform_details,
    ):
        new_pick_needed = True
        # Check if there are any picks with a ptime close to det.time
        close_picks = services.get_picks(
            session,
            self.station_id,
            self.seed_code,
            common_waveform_details["phase"],
            min_time=detection.time - timedelta(seconds=0.1),
            max_time=detection.time + timedelta(seconds=0.1),
        )

        # If there are no close picks, then insert_new_pick = True
        if len(close_picks) == 0:
            return new_pick_needed
        if len(close_picks) > 1:
            self.close_open_pytables()
            raise NotImplementedError(
                "There are multiple close picks in the previous day's data..."
            )

        # If made it to this point, will update or keep an existing pick
        new_pick_needed = False

        # There's only one pick
        pick = close_picks[0]
        prev_data_id = session.get(DLDetection, pick.detid).data_id

        ## Get close waveform or waveform_info.
        if not common_waveform_details["use_pytables"]:
            # Get the waveforms for these picks
            close_wfs = services.get_waveforms(session, pick.id, data_id=prev_data_id)
            prev_inserted_npts = len(close_wfs[0].data)
        else:
            ## This will get waveform info instead of waveform, but they can be treated similarly
            ## Just the data will need to also be grabbed from a pytable.
            close_wfs = services.get_waveform_infos(
                session, pick.id, data_id=prev_data_id, wf_source_id=self.wf_source_id
            )
            prev_inserted_npts = close_wfs[0].duration_samples
        ##

        # Only update the pick and waveforms if the waveforms would have more continuous data available
        if prev_inserted_npts > pick_waveform_details["npts"]:
            return new_pick_needed

        # If so, update the pick time and detection id, everything else should be the same
        pick.ptime = detection.time
        pick.detid = detection.id

        # This will iterate over Waveform if not using pytables and WaveformInfo otherwise
        for wf in close_wfs:
            seed_code = session.get(Channel, wf.chan_id).seed_code
            # Get the appropriate channel index of the data
            chan_ind = self.get_channel_data_index(self.ncomps, seed_code)
            wf.start = pick_waveform_details["wf_start"]
            wf.end = pick_waveform_details["wf_end"]
            pick_cont_data = pick_waveform_details["pick_cont_data"]
            wf.data_id = common_waveform_details["data_id"]
            if not common_waveform_details["use_pytables"]:
                # Get just the channel of interest
                wf.data = pick_cont_data[:, chan_ind].tolist()
            else:
                # Pytables expects a fixed length array
                wf_data = np.zeros(
                    common_waveform_details["expected_array_length"], dtype=np.float32
                )
                wf_start_ind = pick_waveform_details["wf_start_ind"]
                wf_end_ind = pick_waveform_details["wf_end_ind"]
                wf_data[wf_start_ind:wf_end_ind] = pick_cont_data[:, chan_ind]
                # Modify the pytable entry
                storage_dict[wf.chan_id].modify(
                    wf.id, wf_data, wf_start_ind, wf_end_ind
                )
            # TODO: Might want to update this in case the channel switched between days...
            # But I think it makes sense to still assign it to the previous day's channel
            # wf.chan_id = self.channel_info.channel_ids[seed_code]

        return new_pick_needed

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

    def close_open_pytables(self):
        if self.detout_storage_P is not None:
            self.detout_storage_P.close()

        if self.detout_storage_S is not None:
            self.detout_storage_S.close()

        if self.waveform_storage_dict_P is not None:
            for key, stor in self.waveform_storage_dict_P.items():
                stor.close()
        if self.waveform_storage_dict_S is not None:
            for key, stor in self.waveform_storage_dict_S.items():
                stor.close()
