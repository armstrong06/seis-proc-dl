import sys
sys.path.append('..')
from data_processing import extract_events
from detectors.data_processing import make_yboxcar as boxcar
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.file_manager import Write
np.random.seed(49230)

class SplitDetectorData():
    # TODO: I do not know that this is the best way to implement this class

    def __init__(self, window_duration, dt, max_pick_shift, n_duplicate_train, outfile_pref, target_shrinkage=None):
        self.window_duration = window_duration
        self.dt = dt
        self.max_pick_shift = max_pick_shift
        self.n_duplicate_train = n_duplicate_train
        self.outfile_pref = outfile_pref
        self.target_shrinkage = target_shrinkage # make it so that we classifiy at this fraction of central points in the window

        self.signal_train = None
        self.signal_test = None
        self.signal_validate = None

        self.noise_train = None
        self.noise_test = None
        self.noise_validate = None

        self.signal_train_meta = None
        self.signal_test_meta = None
        self.signal_validate_meta = None

        self.noise_train_meta = None
        self.noise_test_meta = None
        self.noise_validate_meta = None
        
        # spliter = Split_Detector_Data(10, 0.01, 250, 2, "path/")
        # spliter.load_signal(h5_filenames, csv_files)
        # split.load_noise(h5_files, csv_files)
        # spliter.split_signal(0.8, 0.5, {extract_params}).process_signal()
        # spliter.split_noise(0.8, 0.5).process_noise()
        # spliter.write_combined_splits()

    def write_combined_datasets(self):
        """Combine and write the split data to files"""
        # TODO: Add way to write when there may not be noise for some reason
        combined = self.__combine_signal_noise(self.signal_train, self.noise_train)
        Write.h5py_file(["X", "Y", "Pick_index"], combined, self.__make_filename("train", "h5"))
        self.signal_train_meta.to_csv(self.__make_filename("train", "csv"), index=False)

        if self.signal_test is not None:
            combined = self.__combine_signal_noise(self.signal_test, self.noise_test)
            Write.h5py_file(["X", "Y", "Pick_index"], combined, self.__make_filename("test", "h5"))
            self.signal_test_meta.to_csv(self.__make_filename("test", "csv"), index=False)

        if self.signal_validate is not None:
            combined = self.__combine_signal_noise(self.signal_validate, self.noise_validate)
            Write.h5py_file(["X", "Y", "Pick_index"], combined, self.__make_filename("validate", "h5"))
            self.signal_validate_meta.to_csv(self.__make_filename("validate", "csv"), index=False)

        if self.noise_train_meta is not None:
            self.noise_train_meta.to_csv(self.__make_filename("train", "csv", is_noise=True), index=False)
            if self.noise_test_meta is not None:
                self.noise_test_meta.to_csv(self.__make_filename("test", "csv", is_noise=True), index=False)
            if self.noise_validate_meta is not None:
                self.noise_validate_meta.to_csv(self.__make_filename("validate", "csv", is_noise=True), index=False)

    def load_signal_data(self, h5_phase_file, meta_csv_file):
        """Read in the signal data"""
        X, Y, meta_df = self.__load_waveform_data(h5_phase_file, meta_csv_file)
        self.signal_train = (X, Y)
        self.signal_train_meta = meta_df

    def load_noise_data(self, h5_phase_file, meta_csv_file=None):
        """Read in the noise data"""
        n_samples_in_window = self.__compute_pad_length()
        X, Y, meta_df = self.__load_waveform_data(h5_phase_file, meta_csv_file, n_samples_in_window)
        self.noise_train = (X, Y)
        self.noise_train_meta = meta_df

    def split_signal(self,
                train_frac = 0.8, # e.g., keep 80 pct of signals for training and 20 pct for validation and testing
                test_frac = 0.5,  # e.g., keep 50 pct of signals for validation and 50 pct of signals for testing
                extract_events_params=None):
        """Split the signal data into train, test, validate datasets"""

        X, Y = self.signal_train
        meta_df = self.signal_train_meta

        # Rename columns if metadata is from STEAD
        if np.isin("source_id", meta_df.columns):
            column_names = np.copy(meta_df.columns.values)
            column_names[np.where(column_names == "source_id")[0][0]] = "evid"
            meta_df.columns = column_names

        # Remove certain events from datasets if necessary
        if extract_events_params is not None:
            X, Y, meta_df = self.__extract_events(extract_events_params)

        assert meta_df.shape[0] == X.shape[0], 'number of waveforms dont match csv'
        assert meta_df.shape[0] == Y.shape[0], 'number of responses dont match csv'
        self.__set_n_signal_waveforms(len(meta_df))

        print("Input data shape:", X.shape, Y.shape)

        print("Sampling signal rows...")
        train_rows, test_rows = self.__sample_on_evid(meta_df, train_frac)

        validate_rows = None
        if (test_frac > 0 and test_frac < 1):
            validate_rows, test_rows = train_test_split(test_rows, train_size = test_frac) 
        else:
            if (test_frac == 1):
                test_rows = np.copy(test_rows)
            else:
                validate_rows = np.copy(test_rows)

        self.signal_train = (X[train_rows, :, :], Y[train_rows])
        self.signal_test_meta = meta_df.iloc[train_rows]

        if validate_rows is not None:
            self.signal_validate = (X[validate_rows, :, :], Y[validate_rows])
            self.signal_validate_meta = meta_df.iloc[validate_rows]

        if test_rows is not None:
            self.signal_test = (X[test_rows, :, :], Y[test_rows])
            self.signal_test_meta = meta_df.iloc[test_rows]

    def process_signal(self):
        """
        Process the signal splits 
        """
        print("Creating signal training dataset...")
        self.signal_train \
            = self.__augment_data(self.signal_train[0], self.signal_train[1], self.n_duplicate_train,
                        target_shrinkage = self.target_shrinkage, lrand=True, for_python=True, meta_for_boxcar=self.signal_train_meta)

        if (self.signal_validate is not None):
            print("Creating signal validation dataset...")
            self.signal_validate \
                = self.__augment_data(self.signal_validate[0], self.signal_validate[1], 1,
                            target_shrinkage = self.target_shrinkage, lrand=True, for_python=True, meta_for_boxcar=self.signal_validate_meta)

        if (self.signal_test is not None):
            print("Creating signal test dataset...")
            self.signal_test \
                = self.__augment_data(self.signal_test[0], self.signal_test[1], 1, 
                            target_shrinkage = self.target_shrinkage, lrand=True, for_python=True, meta_for_boxcar=self.signal_test_meta)

    def split_noise(self, 
                noise_train_frac = 0.8, # e.g., keep 0.8 pct of noise for training and 0.2 pct for validation, 
                test_frac = 0.5,  # e.g., keep 50 pct of signals for validation and 50 pct of signals for testing,
                reduce_stead_noise=False):
        """Split the noise into train, test, and validate datasets"""

        print("Loading noise waveforms...")
        X_noise, Y_noise = self.noise_train
        noise_meta_df = self.noise_train_meta

        print("Input noise shape:", X_noise.shape, Y_noise.shape)

        # TODO: Check that this works
        if reduce_stead_noise:
            X_noise, Y_noise, noise_meta_df = self.__reduce_stead_noise_dataset(noise_meta_df, X_noise, Y_noise)
        
        print("Samping noise rows...")
        n_noise = X_noise.shape[0]
        noise_train_rows, noise_test_rows = train_test_split(np.arange(0, n_noise-1, 1), 
                                                            train_size = noise_train_frac)
        noise_validate_rows = None
        if (test_frac > 0 and test_frac < 1):
            noise_validate_rows, noise_test_rows = train_test_split(noise_test_rows,
                                                                    train_size = test_frac)
        else:
            if (test_frac == 1):
                noise_test_rows = np.copy(noise_test_rows)
            else:
                noise_validate_rows = np.copy(noise_test_rows) 

        if self.noise_train_meta is not None:
            self.noise_train_meta = noise_meta_df.iloc[noise_train_rows]
            if (noise_validate_rows.size > 0):
                self.noise_validate_meta = noise_meta_df.iloc[noise_validate_rows]
            if (noise_test_rows.size > 0):
                self.noise_test_meta = noise_meta_df.iloc[noise_test_rows]

        self.noise_train = (X_noise[noise_train_rows,:,:], Y_noise[noise_train_rows])
        self.noise_test = (X_noise[noise_test_rows,:,:], Y_noise[noise_test_rows])
        self.noise_validate = (X_noise[noise_validate_rows,:,:], Y_noise[noise_validate_rows])

    def process_noise(self):
        """Process the noise splits"""

        print("Creating noise training dataset...")
        noise_train_splits \
            = self.__augment_data(self.noise_train[0], self.noise_train[1], 1,
                        target_shrinkage = self.target_shrinkage, lrand=False, for_python=True)
        noise_train_splits[-1][:] =-1 # Don't care

        noise_validate_splits, noise_test_splits = None, None
        if (self.noise_validate is not None):
            print("Creating noise validation dataset...")
            noise_validate_splits \
                = self.__augment_data(self.noise_validate[0], self.noise_validate[1], 1,
                            target_shrinkage = self.target_shrinkage, lrand=False, for_python=True)
            noise_validate_splits[-1][:] =-1 # Don't care

        if (self.noise_test is not None):
            print("Creating noise test dataset...")
            noise_test_splits \
                = self.__augment_data(self.noise_test[0], self.noise_test[1], 1,
                            target_shrinkage = self.target_shrinkage, lrand=False, for_python=True)
            noise_test_splits[-1][:] =-1 # Don't care

        self.noise_train = noise_train_splits
        self.noise_test = noise_test_splits
        self.noise_validate = noise_validate_splits

    def __set_n_signal_waveforms(self, n_signal_waveforms):
        self.n_signal_waveforms = n_signal_waveforms

    def __make_filename(self, split, file_type, is_noise=False):
        """Creates a filename for different data splits following a common naming scheme"""
        if is_noise:
            return  f'{self.outfile_pref}{split}{self.phase_type}.{self.window_duration}.{self.n_duplicate_train}dup.{file_type}'
        else:
            return  f'{self.outfile_pref}noise_{split}{self.phase_type}.{self.window_duration}.{file_type}'

    def __load_waveform_data(self, h5_file, meta_csv_file, n_samples_in_window=1e6):
        if (type(h5_file) is list):
            self.__join_h5_files(h5_file, n_samples_in_window) # put large number in for n_samples_in_window to keep them all
        else:
            print("Reading:", h5_file)
            h5_file = h5py.File(h5_file, 'r')
            X = np.asarray(h5_file['X'])
            Y = np.asarray(h5_file['Y'])
            h5_file.close()

        if meta_csv_file is None:
            meta_df = None
        elif (type(meta_csv_file) is list):
            meta_df = None
            for meta_csv in meta_csv_file:
                if (meta_df is None):
                    meta_df = pd.read_csv(meta_csv)
                else:
                    meta_df = meta_df.append(pd.read_csv(meta_csv))
                print(meta_csv, meta_df.shape)
        else:
            meta_df = pd.read_csv(meta_csv_file)

        return X, Y, meta_df

    def __compute_pad_length(self):
        n_samples = int(self.window_duration/self.dt + 0.5) + 1
        while (n_samples%16 != 0):
            n_samples = n_samples + 1
        return n_samples

    def __augment_data(self, X, Y, n_duplicate,
                    target_shrinkage = None,
                    lrand = True, for_python=True, meta_for_boxcar=None):
        # Determine the sizes of the data
        print("initial", X.shape, Y.shape)
        n_obs = X.shape[0]
        n_samples = X.shape[1]
        n_comps = X.shape[2]
        # Compute a safe window size for the network architecture
        n_samples_in_window = self.__compute_pad_length()
        if (target_shrinkage is None):
            n_samples_in_target = n_samples_in_window
            start_y = 0
            end_y = 0
        else:
            # TODO: Figure out what target_shrinkage does
            assert target_shrinkage > 0 and target_shrinkage < 1, 'target shrinkage must be in range (0,1)'
            n_samples_in_target = int(target_shrinkage*n_samples_in_window)
            start_y = (n_samples_in_window - n_samples_in_target)//2 
            end_y = n_samples_in_window - n_samples_in_target - start_y
        
        # Define the lags
        #max_lag = int(0.9*n_samples_in_window/2) # TODO: May want to multiply by like 0.95 - worst case is starting half-way into pick window
        
        # If max_pick_shift is a less than 1.0 - use it as a fraction of 1/2 the window length. Otherwise just use the max_lag value
        max_lag = self.max_pick_shift
        if self.max_pick_shift <= 1.0:
            max_lag = int(self.max_pick_shift*n_samples_in_window/2)

        if (lrand):
            lags = np.random.randint(-max_lag, max_lag+1, size=n_obs*n_duplicate)
        else:
            lags = np.zeros(n_obs*n_duplicate, dtype='i')
        
        # Without lags this will center the picks
        start_sample = int(n_samples/2 - n_samples_in_window/2)
        pick_window_index = int(n_samples_in_window/2)
        print(start_sample, n_samples_in_window)

        # Allocate space
        t = np.zeros([n_samples_in_window, n_comps])
        T_index = np.zeros(n_obs*n_duplicate, dtype='int')
        X_new = np.zeros([n_obs*n_duplicate, n_samples_in_window, n_comps])

        if (for_python):
            Y_new = np.zeros([n_obs*n_duplicate, n_samples_in_target, 1])
        else:
            Y_new = np.zeros([n_obs*n_duplicate, n_samples_in_target]) 

        # Sample and augment data
        for idup in range(n_duplicate):
            for iobs in range(n_obs):
                # I added this condition
                if n_samples == 1008:
                    start_index = 0
                else:
                    start_index = start_sample + lags[idup*n_obs+iobs]
                assert start_index >= 0, 'this should never happen'
                end_index = start_index + n_samples_in_window
                
                T_index[idup*n_obs+iobs] = pick_window_index - lags[idup*n_obs+iobs] 

                if X[iobs, start_index:end_index, :].shape[0] != 1008:
                    print('here is a problem')
                t[:,:] = np.copy(X[iobs, start_index:end_index, :])
                
                # min/max rescale
                tscale = np.amax(np.abs(t))
                t = t/tscale
                
                # Copy
                X_new[idup*n_obs+iobs, :, :] = t[:,:]
                if (for_python):
                    # Noise and signal Y had slightly different formats for some reason. Noise does not need 0 index,
                    # signal does
                    #try:
                    Y_new[idup*n_obs+iobs, :, 0] = Y[iobs, start_index+start_y:end_index-end_y]
                    # except:
                    #     try:
                    #         Y_new[idup * n_obs + iobs, :, 0] = Y[iobs, start_index + start_y:end_index - end_y, 0]
                    #     except:
                    #         assert Exception("this didn't work either")
                else:
                    Y_new[idup*n_obs+iobs, :] = Y[iobs, start_index+start_y:end_index-end_y]

        # TODO: I dont know if this will work correct when there are duplicates 
        if meta_for_boxcar is not None:
            Y_new = boxcar.add_boxcar(meta_for_boxcar, {0: 21, 1: 31, 2: 51}, X_new, Y, T_index)

        print("final", X_new.shape, Y_new.shape, T_index.shape)
        return (X_new, Y_new, T_index)

    def __extract_events(self, extract_events_params, meta_df, X, Y):
            print("Extracting events...")
            # Start processing steps...
            extracted_event_meta, kept_event_meta = extract_events.separate_events(meta_df, extract_events_params["bounds"])
            print(X.shape, Y.shape)
            # Grab all the data not including the extracted events
            kept_event_X, kept_event_Y = extract_events.grab_from_h5files(X, Y, kept_event_meta)
            # Grab the data for the extracted events
            extracted_event_X, extracted_event_Y = extract_events.grab_from_h5files(X, Y, extracted_event_meta)
            # Just some sanity checks
            assert X.shape[0] - kept_event_X.shape[0] == extracted_event_X.shape[0]
            assert Y.shape[0] - kept_event_Y.shape[0] == extracted_event_Y.shape[0]
            assert extracted_event_X.shape[0] < kept_event_X.shape[0]
            assert extracted_event_Y.shape[0] < kept_event_Y.shape[0]

            print("Processing extracted events...")
            X_ext, Y_ext, T_ext = self.__augment_data(extracted_event_X[:], extracted_event_Y[:], 1,
                                                            target_shrinkage=None, lrand=True, for_python=True,
                                                            meta_for_boxcar=extracted_event_meta)

            print("Saving extracted event data...")
            extract_events.write_h5file(X_ext, Y_ext, self.__make_filename(extract_events_params["name"], "h5"), T=T_ext)
            extracted_event_meta.to_csv(self.__make_filename(extract_events_params["name"], "df.csv"), index=False)

            return kept_event_X, kept_event_Y, kept_event_meta
 
    def __reduce_stead_noise_dataset(self, noise_meta_df, X_noise, Y_noise):
        print("Filtering STEAD noise rows...")
        print("Original noise shape:", X_noise.shape)
        noise_rows = np.arange(0, len(X_noise), 3)

        if len(noise_rows) < self.n_signal_waveforms:
            rows_to_sample = np.full(len(X_noise), 1)
            rows_to_sample[noise_rows] = 0
            noise_rows = np.append(noise_rows, np.random.choice(np.arange(0, len(X_noise))[np.where(rows_to_sample)], self.n_signal_waveforms-len(noise_rows)))
            noise_rows = np.sort(noise_rows)

        X_noise = X_noise[noise_rows, :, :]
        Y_noise = Y_noise[noise_rows]
        if noise_meta_df is not None:
            noise_meta_df = noise_meta_df.iloc[noise_rows]

        print("New noise shape:", X_noise.shape)

        return X_noise, Y_noise, noise_meta_df

    @staticmethod
    def __sample_on_evid(df, train_frac = 0.8):
        """
        Samples the waveforms on event IDs.  The idea is to prevent the possibility
        of our classifier learning a source time function.
        """
        print("Splitting data...")
        evids = df['evid'].values
        unique_evids = np.unique(evids)
        evids_train, evids_test = train_test_split(unique_evids, train_size=train_frac)

        rows_train = np.where(np.isin(evids, evids_train))[0]
        rows_test = np.where(np.isin(evids, evids_test))[0]

        assert len(rows_test) + len(rows_train) == len(evids), "Did not get all events"

        # Make sure I got all the rows
        frac = len(rows_train)/len(evids) + len(rows_test)/len(evids)
        assert abs(frac - 1.0) < 1.e-10, 'failed to split data'
        # Ensure no test evid is in the training set
        for row_test in rows_test:
            r = np.isin(row_test, rows_train, assume_unique=True)
            assert not r, 'failed to split data'
        print("N_train: %d, Training fraction: %.2f"%(
            len(rows_train), len(rows_train)/len(evids)))
        print("N_test: %d, Testing fraction: %.2f"%(
            len(rows_test), len(rows_test)/len(evids)))
        return rows_train, rows_test #print(rows_train, len(rows_train), len(rows_train)/len(evids))

    @staticmethod
    def __join_h5_files(h5_files, n_samples_in_window):
        X = None
        Y = None
        for h5_f in h5_files:
            print("Reading:", h5_f)
            h5_file = h5py.File(h5_f, 'r')
            print("file length", h5_file["X"].shape)

            file_x = np.asarray(h5_file["X"])
            file_y = np.asarray(h5_file['Y'])
            if file_x.shape[1] > n_samples_in_window:
                file_x = file_x[:, :n_samples_in_window, :]
                if len(file_y.shape) < 3:
                    new_shape = file_y.shape + (1,)
                    file_y = file_y.reshape(new_shape)
                    # tmp=file_x
                file_y = file_y[:, :n_samples_in_window, :]

            print(file_x[0])

            if (X is None):
                X = file_x
                Y = file_y
            else:
                # print(np.where(tmp[0, :, 0] == file_x[0, 0, 0]))
                X = np.concatenate((X, file_x), axis=0)
                Y = np.concatenate((Y, file_y), axis=0)
            h5_file.close()
            print(X.shape)

        return X, Y

    @staticmethod
    def __combine_signal_noise(signal_tuple, noise_tuple):
        combined = []
        for ind in range(3):
            combined.append(np.concatenate((signal_tuple[ind], noise_tuple[ind]), axis=0 ))
        return [combined]

