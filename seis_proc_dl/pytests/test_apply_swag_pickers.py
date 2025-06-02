from seis_proc_dl.apply_to_continuous import apply_swag_pickers
import torch
import numpy as np
import os
import pandas as pd
import h5py
import pytest
from obspy.core.utcdatetime import UTCDateTime as UTC

examples_dir = "/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/seis_proc_dl/pytests/example_files"


class TestMultiSWAGPicker:
    def test_init_P(self):
        sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True)
        assert sp.phase == "P"
        assert sp.device == "cuda:0"

    def test_init_S(self):
        sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=False)
        assert sp.phase == "S"
        assert sp.device == "cuda:0"

    def test_init_cpu(self):
        sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device="cpu")
        assert sp.device == "cpu"

    def test_torch_loader_train(self):
        sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True)
        path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/swag_info"
        loader = sp.torch_loader("p_uuss_train_4s_1dup.h5", path, 256, 3, False)
        assert loader.dataset.data.shape == (336885, 1, 400)
        assert loader.batch_size == 256
        assert loader.num_workers == 3

    def test_torch_loader_cont(self):
        sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True)
        path = "/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/machineLearning/harvestPicks/gcc_build"
        loader = sp.torch_loader("pArrivals.ynpEarthquake.h5", path, 512, 0, False)
        assert loader.dataset.data.shape == (253088, 1, 400)
        assert loader.batch_size == 512
        assert loader.num_workers == 0

    def test_load_swag_ensemble(self):
        sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device="cpu")
        path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"
        model1 = "pPicker_swag60_seed1.pt"
        model2 = "pPicker_swag60_seed2.pt"
        model3 = "pPicker_swag60_seed3.pt"
        ensemble = sp.load_swag_ensemble(
            model1, model2, model3, [1, 2, 3], True, 20, swag_model_dir=path
        )
        assert ensemble[0].seed == 1
        assert ensemble[1].seed == 2
        assert ensemble[2].seed == 3
        assert ensemble[0].model.no_cov_mat == False
        assert ensemble[1].model.no_cov_mat == False
        assert ensemble[2].model.no_cov_mat == False
        assert ensemble[0].model.max_num_models == 20
        assert ensemble[1].model.max_num_models == 20
        assert ensemble[2].model.max_num_models == 20
        assert not torch.equal(
            ensemble[0].model.state_dict()["base.conv1.weight_mean"],
            ensemble[1].model.state_dict()["base.conv1.weight_mean"],
        )
        assert not torch.equal(
            ensemble[1].model.state_dict()["base.conv1.weight_mean"],
            ensemble[2].model.state_dict()["base.conv1.weight_mean"],
        )
        assert not torch.equal(
            ensemble[0].model.state_dict()["base.conv1.weight_mean"],
            ensemble[2].model.state_dict()["base.conv1.weight_mean"],
        )

    def test_get_calibrated_pick_bounds(self):
        sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device="cpu")
        path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"
        file = (
            f"{path}/p_calibration_model_medians_ensemble_IFtrimmed_sklearn1.3.1.joblib"
        )
        lb, ub = sp.get_calibrated_pick_bounds(0.05, 0.95, cal_model_file=file)
        assert lb < 0.05 and lb > 0.0
        assert ub > 0.95 and ub < 1.0

    # def test_apply_picker_P():
    #     sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device='cuda:0')
    #     path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"
    #     file  = f"{path}/pPicker_swag60_seed1.pt"
    #     swag1 = apply_swag_pickers.SwagPicker("PPicker", file, 1,
    #                             cov_mat=True, K=20, device='cuda:0')
    #     path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/swag_info"
    #     data_loader = sp.torch_loader("p_uuss_NGB_4s_1dup.h5",
    #                                 path,
    #                                 10,
    #                                 3,
    #                                 False,
    #                                 10)
    #     assert len(data_loader) == 1
    #     train_loader = sp.torch_loader("p_uuss_train_4s_1dup.h5",
    #                                 path,
    #                                 128,
    #                                 3,
    #                                 False)
    #     new_preds = sp.apply_picker([swag1], data_loader, train_loader, 5)

    #     compare_preds = np.load(f"{path}/p_swag_NGB_uncertainty_40_seed1.npz")['predictions'][:10, :5]

    #     # For some reason, the first predictions do not match
    #     assert np.allclose(new_preds[:, 1:], compare_preds[:, 1:], atol=1e-6)

    def test_calibrate_swag_predictions(self):
        sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device="cpu")
        path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"
        file = (
            f"{path}/p_calibration_model_medians_ensemble_IFtrimmed_sklearn1.3.1.joblib"
        )
        lb_trans, ub_trans = sp.get_calibrated_pick_bounds(
            0.05, 0.95, cal_model_file=file
        )
        summary = sp.calibrate_swag_predictions(
            [0, 10, 0], [1, 0.1, 0.1], lb_trans, ub_trans
        )
        assert summary["arrivalTimeShift"][0] == 0
        assert summary["arrivalTimeShiftSTD"][0] == 1
        assert (
            summary["arrivalTimeShiftLowerBound"][0] < -2
            and summary["arrivalTimeShiftLowerBound"][0] > -4
        )
        assert (
            summary["arrivalTimeShiftUpperBound"][0] > 2
            and summary["arrivalTimeShiftUpperBound"][0] < 4
        )
        assert summary["arrivalTimeShift"][1] == 10
        assert summary["arrivalTimeShiftSTD"][1] == 0.1
        assert (
            summary["arrivalTimeShiftLowerBound"][1] < 9.8
            and summary["arrivalTimeShiftLowerBound"][1] > 9.5
        )
        assert (
            summary["arrivalTimeShiftUpperBound"][1] > 10.2
            and summary["arrivalTimeShiftUpperBound"][1] < 10.5
        )
        assert summary["arrivalTimeShift"][2] == 0
        assert summary["arrivalTimeShiftSTD"][2] == 0.1
        assert (
            summary["arrivalTimeShiftLowerBound"][2] < -0.2
            and summary["arrivalTimeShiftLowerBound"][2] > -0.5
        )
        assert (
            summary["arrivalTimeShiftUpperBound"][2] > 0.2
            and summary["arrivalTimeShiftUpperBound"][2] < 0.5
        )

    # def test_calibrate_swag_predictions_small(self):
    #     sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device="cpu")
    #     path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"
    #     file = (
    #         f"{path}/p_calibration_model_medians_ensemble_IFtrimmed_sklearn1.3.1.joblib"
    #     )
    #     lb_trans, ub_trans = sp.get_calibrated_pick_bounds(
    #         0.05, 0.95, cal_model_file=file
    #     )
    #     summary = sp.calibrate_swag_predictions([0, 0], [1, 0.1], lb_trans, ub_trans)
    #     assert summary["arrivalTimeShift"][1] == 0
    #     assert summary["arrivalTimeShiftSTD"][1] == 0.1
    #     assert summary["arrivalTimeShiftLowerBound"][1] < -0.2
    #     assert summary["arrivalTimeShiftUpperBound"][1] > 0.2

    def test_trim_inner_fence(self):
        sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device="cpu")
        preds = np.random.uniform(0, 1, 25)
        preds[20:] = [-20, 20, -20, 20, 20]
        preds = np.expand_dims(preds, 0)
        median, std = sp.trim_inner_fence(preds)
        assert std < np.std(preds)
        assert median < np.median(preds)

    def test_format_and_save_P(self):
        sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device="cpu")
        pred_summary = {
            "arrivalTimeShift": [0.2],
            "arrivalTimeShiftSTD": [1],
            "arrivalTimeShiftLowerBound": [-2],
            "arrivalTimeShiftUpperBound": [2],
        }
        preds = np.random.uniform(0, 1, 25)
        preds = np.expand_dims(preds, 0)
        outfile_pref = "/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/seis_proc_dl/pytests/example_files"
        region = "ynpEarthquake"
        meta_df = f"{outfile_pref}/test_pick_meta_df.csv"
        sp.format_and_save(meta_df, pred_summary, preds, outfile_pref, region)

        new_df = pd.read_csv(f"{outfile_pref}/corrections.pArrivals.ynpEarthquake.csv")
        assert new_df.shape == (1, 10)
        # assert new_df.shape == (1, 12)
        # assert new_df['correctedArrivalTime'].values[0] == 1349112417.605000 + 0.2
        # assert UTC(new_df['correctedArrivalTime'].values[0]) - UTC(1349112417.605000) == 0.2

        with h5py.File(
            f"{outfile_pref}/corrections.pArrivals.ynpEarthquake.h5", "r"
        ) as f:
            assert np.array_equal(f["X"][:], preds)

        os.remove(f"{outfile_pref}/corrections.pArrivals.ynpEarthquake.csv")
        os.remove(f"{outfile_pref}/corrections.pArrivals.ynpEarthquake.h5")

    def test_format_and_save_P_limit_examples(self):
        sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device="cpu")
        pred_summary = {
            "arrivalTimeShift": [0.2],
            "arrivalTimeShiftSTD": [1],
            "arrivalTimeShiftLowerBound": [-2],
            "arrivalTimeShiftUpperBound": [2],
        }
        preds = np.random.uniform(0, 1, 25)
        preds = np.expand_dims(preds, 0)
        outfile_pref = "/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/seis_proc_dl/pytests/example_files"
        region = "ynpEarthquake"
        meta_df = "/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/machineLearning/harvestPicks/gcc_build/pArrivals.ynpEarthquake.csv"
        sp.format_and_save(
            meta_df, pred_summary, preds, outfile_pref, region, n_meta_rows=1
        )

        new_df = pd.read_csv(f"{outfile_pref}/corrections.pArrivals.ynpEarthquake.csv")
        assert new_df.shape == (1, 10)
        # assert new_df.shape == (1, 12)
        # assert new_df['correctedArrivalTime'].values[0] == 1349112417.605000 + 0.2
        # assert UTC(new_df['correctedArrivalTime'].values[0]) - UTC(1349112417.605000) == 0.2

        with h5py.File(
            f"{outfile_pref}/corrections.pArrivals.ynpEarthquake.h5", "r"
        ) as f:
            assert np.array_equal(f["X"][:], preds)

        os.remove(f"{outfile_pref}/corrections.pArrivals.ynpEarthquake.csv")
        os.remove(f"{outfile_pref}/corrections.pArrivals.ynpEarthquake.h5")

    def test_format_and_save_S(self):
        sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=False, device="cpu")
        pred_summary = {
            "arrivalTimeShift": [0.2],
            "arrivalTimeShiftSTD": [1],
            "arrivalTimeShiftLowerBound": [-2],
            "arrivalTimeShiftUpperBound": [2],
        }
        preds = np.random.uniform(0, 1, 25)
        preds = np.expand_dims(preds, 0)
        outfile_pref = "/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/seis_proc_dl/pytests/example_files"
        region = "ynpEarthquake"
        meta_df = "/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/machineLearning/harvestPicks/gcc_build/sArrivals.ynpEarthquake.csv"
        sp.format_and_save(
            meta_df, pred_summary, preds, outfile_pref, region, n_meta_rows=1
        )

        new_df = pd.read_csv(f"{outfile_pref}/corrections.sArrivals.ynpEarthquake.csv")
        assert new_df.shape == (1, 10)
        # assert new_df.shape == (1, 12)
        # assert new_df['correctedArrivalTime'].values[0] == 1349112417.605000 + 0.2
        # assert UTC(new_df['correctedArrivalTime'].values[0]) - UTC(1349112417.605000) == 0.2

        with h5py.File(
            f"{outfile_pref}/corrections.sArrivals.ynpEarthquake.h5", "r"
        ) as f:
            assert np.array_equal(f["X"][:], preds)

        os.remove(f"{outfile_pref}/corrections.sArrivals.ynpEarthquake.csv")
        os.remove(f"{outfile_pref}/corrections.sArrivals.ynpEarthquake.h5")

    def test_load_data(self):
        pass

    def test_apply_picker_S(self):
        pass


class TestMultiSWAGPickerDB:
    """Just test the functions that don't have anything to do with the database. Database functions are tested in
    test_database_connector
    """

    @pytest.fixture
    def p_ex_paths(self):
        selected_model_path = (
            "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"
        )
        p_cal_file = f"{selected_model_path}/p_calibration_model_medians_ensemble_IFtrimmed_sklearn1.3.1.joblib"

        return selected_model_path, p_cal_file

    @pytest.fixture
    def s_ex_paths(self):
        selected_model_path = (
            "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"
        )
        s_cal_file = f"{selected_model_path}/s_calibration_model_medians_ensemble_IFtrimmed_sklearn1.3.1.joblib"

        return selected_model_path, s_cal_file

    @pytest.fixture
    def init_p_picker(self, p_ex_paths):
        model_path, cal_file = p_ex_paths
        sp = apply_swag_pickers.MultiSWAGPickerDB(
            is_p_picker=True, swag_model_dir=model_path, cal_model_file=cal_file
        )
        return sp

    @pytest.fixture
    def init_s_picker(self, s_ex_paths):
        model_path, cal_file = s_ex_paths
        sp = apply_swag_pickers.MultiSWAGPickerDB(
            is_p_picker=False, swag_model_dir=model_path, cal_model_file=cal_file
        )
        return sp

    def test_init_P(self, p_ex_paths, init_p_picker):
        model_path, cal_file = p_ex_paths
        sp = init_p_picker
        assert sp.phase == "P"
        assert sp.device == "cuda:0"
        assert sp.swag_model_dir == model_path
        assert sp.cal_model_file == cal_file

    def test_init_S(self, s_ex_paths, init_s_picker):
        model_path, cal_file = s_ex_paths
        sp = init_s_picker
        assert sp.phase == "S"
        assert sp.device == "cuda:0"
        assert sp.swag_model_dir == model_path
        assert sp.cal_model_file == cal_file

    def test_process_3c_S(self, init_s_picker):
        sp = init_s_picker
        # 4 columns - time, vertical, north, east
        raw_data = np.loadtxt(
            f"{examples_dir}/ben_data/pickers/cnnThreeComponentS/uu.gzu.eh.zne.01.txt",
            delimiter=",",
        )
        ref_proc_data = np.loadtxt(
            f"{examples_dir}/ben_data/pickers/cnnThreeComponentS/uu.gzu.eh.zne.01.proc.txt",
            delimiter=",",
        )
        assert raw_data.shape == (600, 4)
        assert ref_proc_data.shape == (600, 4)
        # Order ENZ
        idx = [3, 2, 1]
        raw_data = raw_data[:, idx]
        ref_proc_data = ref_proc_data[:, idx]
        assert raw_data.shape == (600, 3)
        assert ref_proc_data.shape == (600, 3)
        proc_data = sp.process_3c_S(raw_data, normalize=False)
        assert proc_data.shape == (600, 3)
        assert np.allclose(proc_data, ref_proc_data, 1.0e-2)
        assert np.max(np.abs(proc_data)) > 1, "data is normalized"

    def test_process_1c_P(self, init_p_picker):
        sp = init_p_picker
        # 2 columns - time, vertical
        raw_data = np.loadtxt(
            f"{examples_dir}/ben_data/pickers/cnnOneComponentP/uu.gzu.ehz.01.txt",
            delimiter=",",
        )
        ref_proc_data = np.loadtxt(
            f"{examples_dir}/ben_data/pickers/cnnOneComponentP/uu.gzu.ehz.01.proc.txt",
            delimiter=",",
        )
        assert raw_data.shape == (400, 2)
        assert ref_proc_data.shape == (400, 2)
        proc_data = sp.process_1c_P(raw_data[:, 1:], normalize=False)
        assert proc_data.shape == (400, 1)
        assert np.allclose(proc_data, ref_proc_data[:, 1:], 1.0e-2)
        assert np.max(np.abs(proc_data)) > 1, "data is normalized"

    def test_process_3c_S_normalize(self, init_s_picker):
        sp = init_s_picker
        # 4 columns - time, vertical, north, east
        raw_data = np.loadtxt(
            f"{examples_dir}/ben_data/pickers/cnnThreeComponentS/uu.gzu.eh.zne.01.txt",
            delimiter=",",
        )
        ref_proc_data = np.loadtxt(
            f"{examples_dir}/ben_data/pickers/cnnThreeComponentS/uu.gzu.eh.zne.01.proc.txt",
            delimiter=",",
        )
        assert raw_data.shape == (600, 4)
        assert ref_proc_data.shape == (600, 4)
        # Order ENZ
        idx = [3, 2, 1]
        raw_data = raw_data[:, idx]
        ref_proc_data = ref_proc_data[:, idx]
        assert raw_data.shape == (600, 3)
        assert ref_proc_data.shape == (600, 3)
        ref_proc_data = sp.normalize_example(ref_proc_data)
        proc_data = sp.process_3c_S(raw_data, normalize=True)
        assert proc_data.shape == (600, 3)
        assert np.allclose(proc_data, ref_proc_data, 1.0e-2)
        assert np.all(np.max(np.abs(proc_data), axis=0) == 1), "data is not normalized"

    def test_process_1c_P_normalize(self, init_p_picker):
        sp = init_p_picker
        # 2 columns - time, vertical
        raw_data = np.loadtxt(
            f"{examples_dir}/ben_data/pickers/cnnOneComponentP/uu.gzu.ehz.01.txt",
            delimiter=",",
        )
        ref_proc_data = np.loadtxt(
            f"{examples_dir}/ben_data/pickers/cnnOneComponentP/uu.gzu.ehz.01.proc.txt",
            delimiter=",",
        )
        assert raw_data.shape == (400, 2)
        assert ref_proc_data.shape == (400, 2)
        ref_proc_data = sp.normalize_example(ref_proc_data)
        proc_data = sp.process_1c_P(raw_data[:, 1:], normalize=True)
        assert proc_data.shape == (400, 1)
        assert np.allclose(proc_data, ref_proc_data[:, 1:], 1.0e-2)
        assert np.max(np.abs(proc_data)) == 1, "data is not normalized"

    def test_get_lb_ub_from_percent(self, init_p_picker):
        sp = init_p_picker

        lb, ub = sp._get_lb_ub_from_percent(68)
        assert lb == 0.16
        assert ub == 0.84

        lb, ub = sp._get_lb_ub_from_percent(90)
        assert lb == 0.05
        assert ub == 0.95

    def test_get_calibrated_pick_bounds_percent(self, init_p_picker):
        sp = init_p_picker
        lb, ub = sp.get_calibrated_pick_bounds_percent(90)
        # Rough check on bounds.
        # The lb should be smaller than without calibration but greater than 0
        assert lb < 0.05 and lb > 0.0
        # The ub should be greater than without calibration but less than 1
        assert ub > 0.95 and ub < 1.0

    def test_get_multiple_cis(self, init_p_picker):
        sp = init_p_picker
        cis_summary = sp.get_multiple_cis([0], [1], [68, 90])
        assert len(list(cis_summary.keys())) == 2
        assert cis_summary[68]["arrivalTimeShift"][0] == 0
        assert cis_summary[68]["arrivalTimeShiftSTD"][0] == 1
        assert (
            cis_summary[68]["arrivalTimeShiftLowerBound"][0] < -1
            and cis_summary[68]["arrivalTimeShiftLowerBound"][0] > -2
        )
        assert (
            cis_summary[68]["arrivalTimeShiftUpperBound"][0] > 1
            and cis_summary[68]["arrivalTimeShiftUpperBound"][0] < 2
        )
        assert cis_summary[90]["arrivalTimeShift"][0] == 0
        assert cis_summary[90]["arrivalTimeShiftSTD"][0] == 1
        assert cis_summary[90]["arrivalTimeShiftLowerBound"][0] < -2
        assert cis_summary[90]["arrivalTimeShiftUpperBound"][0] > 2

    def test_trim_dists(self, init_p_picker):
        sp = init_p_picker
        preds = np.random.normal(size=(2, 151))
        preds[:, -1] = 10
        org_stds = np.std(preds, axis=1)
        org_means = np.mean(preds, axis=1)
        org_medians = np.median(preds, axis=1)

        trim_results = sp.trim_dists(preds)

        for key, item in trim_results.items():
            assert len(item) == 2

        for i in range(2):
            assert trim_results["trim_std"][i] < org_stds[i]
            assert trim_results["trim_mean"][i] < org_means[i]
            assert (
                abs(trim_results["trim_median"][i] - org_medians[i]) < 0.05
            ), "medians should be close"
            assert trim_results["if_low"][i] < trim_results["if_high"][i]
            assert trim_results["if_low"][i] < 0
            assert trim_results["if_high"][i] > 1

    def test_get_summary_stats(self, init_p_picker):
        sp = init_p_picker
        preds = np.random.normal(size=(2, 151))
        preds[:, -1] = 10
        org_stds = np.std(preds, axis=1)
        org_means = np.mean(preds, axis=1)
        org_medians = np.median(preds, axis=1)

        summary_dict = sp.get_summary_stats(preds)

        assert len(summary_dict.keys()) == 8

        for key, item in summary_dict.items():
            assert len(item) == 2

        for i in range(2):
            assert summary_dict["std"][i] == org_stds[i]
            assert summary_dict["mean"][i] == org_means[i]
            assert summary_dict["median"][i] == org_medians[i]
            assert summary_dict["trim_std"][i] < org_stds[i]
            assert summary_dict["trim_mean"][i] < org_means[i]
            assert (
                abs(summary_dict["trim_median"][i] - org_medians[i]) < 0.05
            ), "medians should be close"
            assert summary_dict["if_low"][i] < summary_dict["if_high"][i]
            assert summary_dict["if_low"][i] < 0
            assert summary_dict["if_high"][i] > 1


if __name__ == "__main__":
    TestMultiSWAGPicker.test_format_and_save_S()
