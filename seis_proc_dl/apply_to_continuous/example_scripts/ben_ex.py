#!/usr/bin/env python
import sys
sys.path.append("/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/mlmodels/intel_cpu_build")
import pyuussmlmodels
import numpy as np
import os

PREF = "/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/mlmodels/intel_cpu_build"

def read_time_series(file_name): 
    assert os.path.exists(file_name), '{} does not exist'.format(file_name)
    infl = open(file_name, 'r')
    cdat = infl.read()
    infl.close()
    cdat = cdat.split('\n')
    n = len(cdat) - 1
    t = np.zeros(n)
    x = np.zeros(n)
    for i in range(n):
        ti, xi = cdat[i].split(',')
        t[i] = float(ti)
        x[i] = float(xi)
    return t, x

def read_time_series_3c(file_name): 
    assert os.path.exists(file_name), '{} does not exist'.format(file_name)
    infl = open(file_name, 'r')
    cdat = infl.read()
    infl.close()
    cdat = cdat.split('\n')
    n = len(cdat) - 1 
    t = np.zeros(n)
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    for i in range(n):
        ti, xi, yi, zi = cdat[i].split(',')
        t[i] = float(ti)
        x[i] = float(xi)
        y[i] = float(yi)
        z[i] = float(zi)
    return t, x, y, z

def test_p_one_component_p_detector_preprocessing():
   print("Performing P 1C UNet preprocessing test...")
   t, x = read_time_series(f'{PREF}/data/detectors/uNetOneComponentP/PB.B206.EHZ.zrunet_p.txt')
   t, y_ref = read_time_series(f'{PREF}/data/detectors/uNetOneComponentP/PB.B206.EHZ.PROC.zrunet_p.txt')
   assert len(x) == len(y_ref)
   assert len(x) == 360000
   sampling_rate = round(1./(t[1] - t[0]))
   assert sampling_rate == 100, 'sampling rate should be 100 Hz'
   preprocessor = pyuussmlmodels.Detectors.UNetOneComponentP.Preprocessing()
   assert preprocessor.target_sampling_rate == 100, 'target sampling rate should be 100 Hz'
   y = preprocessor.process(x, sampling_rate = sampling_rate)
   assert max(abs(y - y_ref)) < 1.e-1

def test_p_one_component_p_detector():
   try:
        inference = pyuussmlmodels.Detectors.UNetOneComponentP.Inference()
   except:
        print("Not compiled with inference")
        return
   print("Performing P 1C UNet inference test...")
   t, vertical_ref = read_time_series(f'{PREF}/data/detectors/uNetOneComponentP/PB.B206.EHZ.PROC.zrunet_p.txt')
   t, p_ref        = read_time_series(f'{PREF}/data/detectors/uNetOneComponentP/PB.B206.P_PROBA.SLIDING.txt')
   inference.load(f'{PREF}/../detectors/uNetOneComponentP/models/detectorsUNetOneComponentP.onnx',
                  pyuussmlmodels.Detectors.UNetOneComponentP.ModelFormat.ONNX)
   assert inference.is_initialized
   p_signal = inference.predict_probability(vertical_ref, use_sliding_window = True)
   assert len(p_signal) == len(p_ref)
   assert max(abs(p_signal - p_ref)) < 1.e-4

def test_p_three_component_p_detector_preprocessing():
   print("Performing P 3C UNet preprocessing test...")
   t, vertical = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentP/PB.B206.EHZ.zrunet_p.txt') 
   t, north    = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentP/PB.B206.EH1.zrunet_p.txt')
   t, east     = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentP/PB.B206.EH2.zrunet_p.txt')
   t, vertical_ref = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentP/PB.B206.EHZ.PROC.zrunet_p.txt') 
   t, north_ref    = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentP/PB.B206.EH1.PROC.zrunet_p.txt')
   t, east_ref     = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentP/PB.B206.EH2.PROC.zrunet_p.txt')
   assert len(vertical) == 360000
   assert len(vertical) == len(north)
   assert len(vertical) == len(east) 
   assert len(vertical) == len(vertical_ref)
   assert len(vertical) == len(north_ref)
   assert len(vertical) == len(east_ref)
   sampling_rate = round(1./(t[1] - t[0]))
   assert sampling_rate == 100, 'sampling rate should be 100 Hz'
   preprocessor = pyuussmlmodels.Detectors.UNetThreeComponentP.Preprocessing()
   assert preprocessor.target_sampling_rate == 100, 'target sampling rate should be 100 Hz'
   vertical_proc, north_proc, east_proc \
       = preprocessor.process(vertical, north, east, sampling_rate = sampling_rate)
   assert max(abs(vertical_proc - vertical_ref)) < 1.e-1
   assert max(abs(north_proc - north_ref)) < 1.e-1
   assert max(abs(east_proc - east_ref)) < 1.e-1

def test_p_three_component_p_detector():
   try:
        inference = pyuussmlmodels.Detectors.UNetThreeComponentP.Inference()
   except:
        print("Not compiled with inference")
        return
   print("Performing P 3C UNet inference test...")
   t, vertical_ref = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentP/PB.B206.EHZ.PROC.zrunet_p.txt')
   t, north_ref    = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentP/PB.B206.EH1.PROC.zrunet_p.txt')
   t, east_ref     = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentP/PB.B206.EH2.PROC.zrunet_p.txt')
   t, p_ref        = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentP/PB.B206.P_PROBA.SLIDING.txt')
   inference.load(f'{PREF}/../detectors/uNetThreeComponentP/models/detectorsUNetThreeComponentP.onnx', 
                  pyuussmlmodels.Detectors.UNetThreeComponentP.ModelFormat.ONNX)
   assert inference.is_initialized
   p_signal = inference.predict_probability(vertical_ref, north_ref, east_ref, use_sliding_window = True)
   assert len(p_signal) == len(p_ref)
   assert max(abs(p_signal - p_ref)) < 1.e-4

def test_p_three_component_p_detector_nosliding():
   try:
        inference = pyuussmlmodels.Detectors.UNetThreeComponentP.Inference()
   except:
        print("Not compiled with inference")
        return
   print("Performing P 3C UNet inference test...")
   t, vertical_ref = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentP/PB.B206.EHZ.PROC.zrunet_p.txt')
   t, north_ref    = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentP/PB.B206.EH1.PROC.zrunet_p.txt')
   t, east_ref     = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentP/PB.B206.EH2.PROC.zrunet_p.txt')
   t, p_ref        = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentP/PB.B206.P_PROBA.SLIDING.txt')
   inference.load(f'{PREF}/../detectors/uNetThreeComponentP/models/detectorsUNetThreeComponentP.onnx', 
                  pyuussmlmodels.Detectors.UNetThreeComponentP.ModelFormat.ONNX)
   print(vertical_ref.shape, north_ref.shape, east_ref.shape)
   vertical_ref1, north_ref1, east_ref1 = vertical_ref[0:1008], north_ref[0:1008], east_ref[0:1008]
   print(vertical_ref1.shape, north_ref1.shape, east_ref1.shape)
   p_ref1 = p_ref[0:1008]
   assert inference.is_initialized
   p_signal = inference.predict_probability(vertical_ref1, north_ref1, east_ref1, use_sliding_window = False)
   assert len(p_signal) == len(p_ref1)
   assert max(abs(p_signal - p_ref1)) < 1.e-4
   
def test_p_three_component_s_detector_preprocessing():
   print("Performing S 3C UNet preprocessing test...")
   t, vertical = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentS/PB.B206.EHZ.zrunet_s.txt') 
   t, north    = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentS/PB.B206.EH1.zrunet_s.txt')
   t, east     = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentS/PB.B206.EH2.zrunet_s.txt')
   t, vertical_ref = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentS/PB.B206.EHZ.PROC.zrunet_s.txt') 
   t, north_ref    = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentS/PB.B206.EH1.PROC.zrunet_s.txt')
   t, east_ref     = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentS/PB.B206.EH2.PROC.zrunet_s.txt')
   assert len(vertical) == 360000
   assert len(vertical) == len(north)
   assert len(vertical) == len(east) 
   assert len(vertical) == len(vertical_ref)
   assert len(vertical) == len(north_ref)
   assert len(vertical) == len(east_ref)
   sampling_rate = round(1./(t[1] - t[0]))
   assert sampling_rate == 100, 'sampling rate should be 100 Hz'
   preprocessor = pyuussmlmodels.Detectors.UNetThreeComponentS.Preprocessing()
   assert preprocessor.target_sampling_rate == 100, 'target sampling rate should be 100 Hz'
   vertical_proc, north_proc, east_proc \
       = preprocessor.process(vertical, north, east, sampling_rate = sampling_rate)
   assert max(abs(vertical_proc - vertical_ref)) < 1.e-1
   assert max(abs(north_proc - north_ref)) < 1.e-1
   assert max(abs(east_proc - east_ref)) < 1.e-1

def test_p_one_component_pick_regressor_preprocessing():
   print("Performing P 1C picker preprocessing test...")
   t, x = read_time_series(f'{PREF}/data/pickers/cnnOneComponentP/uu.gzu.ehz.01.txt')
   t, y_ref = read_time_series(f'{PREF}/data/pickers/cnnOneComponentP/uu.gzu.ehz.01.proc.txt')
   assert len(x) == 400 
   assert len(y_ref) == 400 
   sampling_rate = round(1./(t[1] - t[0]))
   assert sampling_rate == 100, 'sampling rate should be 100Hz'
   preprocessor = pyuussmlmodels.Pickers.CNNOneComponentP.Preprocessing()
   assert preprocessor.target_sampling_rate == 100, 'target sampling rate should be 100 Hz'
   assert abs(preprocessor.target_sampling_rate - 1./preprocessor.target_sampling_period) < 1.e-8
   y = preprocessor.process(x, sampling_rate = sampling_rate)
   assert max(abs(y - y_ref)) < 1.e-3

def test_s_three_component_pick_regressor_preprocessing():
   print("Performing S 3C picker preprocessing test...")
   t, vertical, north, east = read_time_series_3c(f'{PREF}/data/pickers/cnnThreeComponentS/uu.gzu.eh.zne.01.txt')
   t, vertical_ref, north_ref, east_ref = read_time_series_3c(f'{PREF}/data/pickers/cnnThreeComponentS/uu.gzu.eh.zne.01.proc.txt')
   assert len(vertical) == 600 
   assert len(vertical_ref) == 600 
   sampling_rate = round(1./(t[1] - t[0]))
   assert sampling_rate == 100, 'sampling rate should be 100Hz'
   preprocessor = pyuussmlmodels.Pickers.CNNThreeComponentS.Preprocessing()
   assert preprocessor.target_sampling_rate == 100, 'target sampling rate should be 100 Hz'
   assert abs(preprocessor.target_sampling_rate - 1./preprocessor.target_sampling_period) < 1.e-8
   v_proc, n_proc, e_proc \
       = preprocessor.process(vertical, north, east, sampling_rate = sampling_rate)
   assert max(abs(v_proc - vertical_ref)) < 1.e-2
   assert max(abs(n_proc - north_ref)) < 1.e-2
   assert max(abs(e_proc - east_ref)) < 1.e-2

def test_s_three_component_s_detector():
   try:
        inference = pyuussmlmodels.Detectors.UNetThreeComponentS.Inference()
   except:
        print("Not compiled with inference")
        return 
   print("Performing S 3C UNet inference test...")
   t, vertical_ref = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentS/PB.B206.EHZ.PROC.zrunet_s.txt')
   t, north_ref    = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentS/PB.B206.EH1.PROC.zrunet_s.txt')
   t, east_ref     = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentS/PB.B206.EH2.PROC.zrunet_s.txt')
   t, p_ref        = read_time_series(f'{PREF}/data/detectors/uNetThreeComponentS/PB.B206.S_PROBA.SLIDING.txt')
   inference.load(f'{PREF}/../detectors/uNetThreeComponentS/models/detectorsUNetThreeComponentS.onnx', 
                  pyuussmlmodels.Detectors.UNetThreeComponentS.ModelFormat.ONNX)
   assert inference.is_initialized
   p_signal = inference.predict_probability(vertical_ref, north_ref, east_ref, use_sliding_window = True)
   assert len(p_signal) == len(p_ref)
   assert max(abs(p_signal - p_ref)) < 1.e-4

def test_p_one_component_first_motion_classifier_preprocessing():
   print("Performing P 1C first motion preprocessing test...")
   t, x = read_time_series(f'{PREF}/data/firstMotionClassifiers/cnnOneComponentP/uu.gzu.ehz.01.txt')
   t, y_ref = read_time_series(f'{PREF}/data/firstMotionClassifiers/cnnOneComponentP/uu.gzu.ehz.01.proc.txt')
   assert len(x) == 400
   assert len(y_ref) == 400
   sampling_rate = round(1./(t[1] - t[0]))
   assert sampling_rate == 100, 'sampling rate should be 100Hz'
   preprocessor = pyuussmlmodels.FirstMotionClassifiers.CNNOneComponentP.Preprocessing()
   assert preprocessor.target_sampling_rate == 100, 'target sampling rate should be 100 Hz'
   assert abs(preprocessor.target_sampling_rate - 1./preprocessor.target_sampling_period) < 1.e-8
   y = preprocessor.process(x, sampling_rate = sampling_rate)
   assert max(abs(y - y_ref)) < 1.e-3



if __name__ == "__main__":
   #test_p_one_component_p_detector_preprocessing()
   #test_p_three_component_p_detector_preprocessing()
   #test_p_three_component_s_detector_preprocessing() 
   #test_p_one_component_pick_regressor_preprocessing()
   #test_s_three_component_pick_regressor_preprocessing()
   #test_p_one_component_first_motion_classifier_preprocessing()
    
   # Inference
   test_p_three_component_p_detector_nosliding()
   #test_s_three_component_s_detector()
   #test_p_one_component_p_detector()

