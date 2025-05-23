import sys
sys.path.append("/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis-proc-dl")
from data_processing.split_data_detectors import SplitDetectorData
import numpy as np
import pandas as pd
from utils.file_manager import Write

window_duration = 10.0
n_duplicate_train = 1
dt = 0.01  # Sampling period (seconds)
train_frac = 0.8
noise_train_frac = 0.8
test_frac = 0.5
max_pick_shift = 250
phase_type ='S'

boxcar_widths={0: 51, 1: 51, 2: 51}

pref = '/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive'
ys_noise_h5_filename = f'{pref}/noise_ENZ/allNoiseYellowstoneWaveformsPermuted.h5'
magna_noise_h5_filename = f'{pref}/noise_ENZ/allNoiseMagnaWaveformsPermuted.P.10s.h5'

pref = f'{pref}/sDetector'
outdir = "constant_bc_width/NGB_S_resampled_nodups"

# For NGB events - Don't need these for STEAD data
# Remove events within these bounds
lat_min = 44.680
lat_max = 44.870
lon_max = 110.850
lon_min = 110.640 #110.640 was old, smaller bound
# Larger location grid bounds
# lat_min = 44.35
# lat_max = 45.0
# lon_max = 111.46
# lon_min = 110.1
date_min = 130901  # YYMMDD
date_max = 140630  # YYMMDD
extracted_name =  "NGB"

################ End Set Params ####################

assert lon_min < lon_max, "lon min must be less than max"
assert lat_min < lat_max, "lat min must be less than max"
assert date_min < date_max, "date min must be less than max"

bounds = {"lat_min": lat_min,
          "lat_max": lat_max,
          "lon_max": lon_max,
          "lon_min": lon_min,
          "date_min": date_min,
          "date_max": date_max}

extract_events_params = {"bounds":bounds, "name":extracted_name}

## Current Earthquakes
h5_filename = f'{pref}/S_current_earthquake_catalog.h5'
meta_file = f'{pref}/S_current_earthquake_catalog.csv'
outpref = f"{pref}/{outdir}/currenteq."

spliter = SplitDetectorData(window_duration, dt, max_pick_shift, n_duplicate_train, phase_type, boxcar_widths, outfile_pref=outpref)
spliter.load_signal_data(h5_filename, meta_file)
spliter.split_signal(train_frac, test_frac, extract_events_params=extract_events_params)
spliter.process_signal()

spliter.load_noise_data([ys_noise_h5_filename, magna_noise_h5_filename])
spliter.split_noise(noise_train_frac, test_frac)
spliter.process_noise()

spliter.write_combined_datasets()
ceq_train,ceq_test,ceq_validate = spliter.return_signal()
ceq_train_df, ceq_test_df, ceq_validate_df = spliter.return_signal_meta()
ceq_train_noise, ceq_test_noise, ceq_validate_noise = spliter.return_noise()
ceq_train_noise_df, ceq_test_noise_df, ceq_validate_noise_df = spliter.return_noise_meta()

# Blast catalog
h5_filename = f'{pref}/S_current_blast_catalog_3c.h5'
meta_file = f'{pref}/S_current_blast_catalog_3c.csv'
spliter = SplitDetectorData(window_duration, dt, max_pick_shift, 1, phase_type, boxcar_widths)
spliter.load_signal_data(h5_filename, meta_file, min_training_quality=0.5)
spliter.split_signal(0.4, 0.96, extract_events_params=None)
spliter.process_signal()
cbl_train,cbl_test,cbl_validate = spliter.return_signal()
cbl_train_df, cbl_test_df, cbl_validate_df = spliter.return_signal_meta()

# Historical Earthquakes
h5_filename = f'{pref}/S_historical_earthquake_catalog.h5'
meta_file = f'{pref}/S_historical_earthquake_catalog.csv'
outpref = f"{pref}/{outdir}/combined."
spliter = SplitDetectorData(window_duration, dt, max_pick_shift, 1, phase_type, boxcar_widths, outfile_pref=outpref)
spliter.load_signal_data(h5_filename, meta_file, min_training_quality=0.75)
spliter.split_signal(0.2, 0.98, extract_events_params=None)
spliter.process_signal()
heq_train,heq_test,heq_validate = spliter.return_signal()
heq_train_df, heq_test_df, heq_validate_df = spliter.return_signal_meta()

def concate_splits(ceq_split, ceq_split_df, cbl_split, cbl_split_df, heq_split, heq_split_df, noise_split=None, noise_split_df=None):
    X = np.concatenate([ceq_split[0], cbl_split[0], heq_split[0]])
    y = np.concatenate([ceq_split[1], cbl_split[1], heq_split[1]])
    T = np.concatenate([ceq_split[2], cbl_split[2], heq_split[2]])
    df = pd.concat([ceq_split_df, cbl_split_df, heq_split_df])
    if noise_split is not None:
        X = np.concatenate([X, noise_split[0]])
        y = np.concatenate([y, noise_split[1]])
        T = np.concatenate([T, noise_split[2]])
    if noise_split_df is not None:
        df = pd.concat([df, noise_split_df])

    print("Current earthquake duplicates", n_duplicate_train)
    print("Current earthquakes:", len(ceq_split_df), (len(ceq_split_df)/len(df))*100)
    print("Historical earthquakes:", len(heq_split_df), (len(heq_split_df)/len(df))*100)
    print("Current Quarry Blasts:", len(cbl_split_df), (len(cbl_split_df)/len(df))*100)

    return [X, y, T], df

train_data, train_df = concate_splits(ceq_train, ceq_train_df, cbl_train, cbl_train_df,
                                            heq_train, heq_train_df, ceq_train_noise)

test_data, test_df = concate_splits(ceq_test, ceq_test_df, cbl_test, cbl_test_df,
                                            heq_test, heq_test_df, ceq_test_noise)

validate_data, validate_df = concate_splits(ceq_validate, ceq_validate_df, cbl_validate, cbl_validate_df,
                                            heq_validate, heq_validate_df, ceq_validate_noise)

Write.h5py_file(["X", "Y", "Pick_index"], train_data, spliter.make_filename("train", "h5"))
Write.h5py_file(["X", "Y", "Pick_index"], test_data, spliter.make_filename("test", "h5"))
Write.h5py_file(["X", "Y", "Pick_index"], validate_data, spliter.make_filename("validate", "h5"))

train_df.to_csv(spliter.make_filename("train", "csv"), index=False)
test_df.to_csv(spliter.make_filename("test", "csv"), index=False)
validate_df.to_csv(spliter.make_filename("validate", "csv"), index=False)
