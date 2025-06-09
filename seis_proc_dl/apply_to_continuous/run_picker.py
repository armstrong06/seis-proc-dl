import os
import argparse
import json
import time
from datetime import datetime
from seis_proc_dl.apply_to_continuous import apply_swag_pickers

### Handle user inputs for picker ###
argParser = argparse.ArgumentParser()
# argParser.add_argument("-p", "--is_p", type=bool, help='True if P arrival, False if S')
argParser.add_argument(
    "-p",
    "--is_p",
    action=argparse.BooleanOptionalAction,
    help="True if P arrival, False if S",
)
argParser.add_argument(
    "--dur",
    type=int,
    help="The duration of the waveform snippets (in samples)",
    default=400,
)
argParser.add_argument(
    "--pad",
    type=int,
    help="The duration (in samples) of padding to add to each end of the waveform window when processing",
    default=100,
)
argParser.add_argument(
    "--start",
    type=str,
    help="The start date in yyyy-mm-dd format",
)
argParser.add_argument(
    "--end",
    type=str,
    help="The end date in yyyy-mm-dd format",
)
argParser.add_argument(
    "-ci",
    "--cred_ivls",
    type=int,
    nargs="+",
    help="list of credible interval percentages to store",
    default=[68, 90],
)
argParser.add_argument(
    "--sources",
    type=str,
    nargs="+",
    help="list of waveform_sources to use. The order defines the source priority and only the top one will be used.",
    default=["extract-dailyContData"],
)
argParser.add_argument(
    "--repicker_name",
    type=str,
    help="repicker method name",
)
argParser.add_argument(
    "--repicker_desc",
    type=str,
    help="repicker method description",
)
argParser.add_argument(
    "-s1", "--swag_model1", type=str, help="first swag model file name"
)
argParser.add_argument(
    "-s2", "--swag_model2", type=str, help="second swag model file name"
)
argParser.add_argument(
    "-s3", "--swag_model3", type=str, help="third swag model file name"
)
argParser.add_argument(
    "-tf", "--train_file", type=str, help="training data file (no path)"
)
argParser.add_argument(
    "-cm", "--cal_file", type=str, help="calibration model file name"
)
argParser.add_argument(
    "--cal_name",
    type=str,
    help="calibration method name",
)
argParser.add_argument(
    "--cal_desc",
    type=str,
    help="calibration method description",
)
argParser.add_argument(
    "--cal_loc_type",
    type=str,
    help="The scale type used in the calibration",
    default="trim_median",
)
argParser.add_argument(
    "--cal_scale_type",
    type=str,
    help="The scale type used in the calibration",
    default="trim_std",
)
argParser.add_argument(
    "-n", "--N", type=int, help="Number of samples to draw from each model"
)
argParser.add_argument(
    "-tb",
    "--train_batchsize",
    type=int,
    help="batch size for the training data bn_update",
    default=512,
)
argParser.add_argument(
    "-db", "--data_batchsize", type=int, help="batch size for the new data", default=512
)
argParser.add_argument(
    "-tw",
    "--train_n_workers",
    type=int,
    help="number of workers for the training data loader",
    default=4,
)
argParser.add_argument(
    "-dw",
    "--data_n_workers",
    type=int,
    help="number of workers for the data loader",
    default=4,
)
argParser.add_argument(
    "-d", "--device", type=str, help="device to use", default="cuda:0"
)
argParser.add_argument(
    "-m",
    "--model_path",
    type=str,
    help="path to the stored models",
    default="/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models",
)
argParser.add_argument(
    "-s",
    "--seeds",
    type=int,
    nargs="+",
    help="intial seeds for the models",
    default=[1, 2, 3],
)
argParser.add_argument(
    "-c",
    "--cov_mat",
    action=argparse.BooleanOptionalAction,
    help="swag cov_mat",
    default=True,
)
argParser.add_argument(
    "-k", "--K", type=int, help="max number of swag models stored", default=20
)
argParser.add_argument(
    "-tp",
    "--train_path",
    type=str,
    help="training data path",
    default="/uufs/chpc.utah.edu/common/home/koper-group3/alysha/swag_info",
)
argParser.add_argument(
    "-st",
    "--shuffle_train",
    action=argparse.BooleanOptionalAction,
    help="shuffle the training data for bn_update",
    default=False,
)
argParser.add_argument(
    "--save_args",
    action=argparse.BooleanOptionalAction,
    help="write args to json file in the working dir",
    default=True,
)
args = argParser.parse_args()

assert os.path.exists(
    os.path.join(args.model_path, args.swag_model1)
), "SWAG model 1 path incorrect"
assert os.path.exists(
    os.path.join(args.model_path, args.swag_model2)
), "SWAG model 2 path incorrect"
assert os.path.exists(
    os.path.join(args.model_path, args.swag_model3)
), "SWAG model 3 path incorrect"
assert os.path.exists(
    os.path.join(args.model_path, args.cal_file)
), "Calibration file path incorrect"
assert os.path.exists(
    os.path.join(args.train_path, args.train_file)
), "Train file path incorrect"
assert args.device in ["cpu", "cuda:0", "cuda:1"], "invalid device type"
assert len(args.seeds) == 3, "Incorrect number of seeds"

start_date = datetime.strptime(args.start, "%Y-%m-%d")
end_date = datetime.strptime(args.end, "%Y-%m-%d")

assert end_date > start_date, "end_date should be larger than start_date"
assert args.dur > 0, "waveform duration (--dur) must be > 0"
assert args.pad >= 0, "pad should be >= 0"

for ci in args.cred_ivls:
    assert ci > 0 and ci < 100, "credible interval must be > 0 and < 100"

print("is_p:", args.is_p)
print("cov_mat:", args.cov_mat)
print("shuffle_train:", args.shuffle_train)
print("save_args:", args.save_args)
if args.save_args:
    phase = "P"
    if not args.is_p:
        phase = "S"
    args_dir = "./args"
    if not os.path.exists(args_dir):
        try:
            os.makedirs(args_dir)
        except:
            pass
    arg_outfile = os.path.join(args_dir, f"args.{phase}.{args.start}.{args.end}.json")
    print("writing", arg_outfile)
    with open(arg_outfile, "w") as f:
        json.dump(args.__dict__, f, indent=2)

#############
st = time.time()
# Initialize the picker
sp = apply_swag_pickers.MultiSWAGPickerDB(
    is_p_picker=args.is_p,
    swag_model_dir=args.model_path,
    cal_model_file=os.path.join(args.model_path, args.cal_file),
    device=args.device,
)
sp.start_db_conn(
    repicker_dict={"name": args.repicker_name, "desc": args.repicker_desc},
    cal_dict={"name": args.cal_name, "desc": args.cal_desc},
    cal_loc_type=args.cal_loc_type,
    cal_scale_type=args.cal_scale_type,
)
# Load the new estimated picks
ids, data_loader = sp.torch_loader_from_db(
    n_samples=args.dur,
    batch_size=args.data_batchsize,
    num_workers=args.data_n_workers,
    start_date=start_date,
    end_date=end_date,
    wf_source_list=args.sources,
    padding=args.pad
)

# Load the training data for bn_updates:q
train_loader = sp.torch_loader(
    args.train_file,
    args.train_path,
    args.train_batchsize,
    args.train_n_workers,
    shuffle=args.shuffle_train,
)

# Load the MultiSWAG ensemble
ensemble = sp.load_swag_ensemble(
    args.swag_model1,
    args.swag_model2,
    args.swag_model3,
    args.seeds,
    args.cov_mat,
    args.K,
)
# Get the posterior predictive distributions for each pick
new_preds = sp.apply_picker(ensemble, data_loader, train_loader, args.N)
# Compute the summary stats and CIs and save results in the database
sp.calibrate_and_save(
    pick_source_ids=ids,
    ensemble_outputs=new_preds,
    ci_percents=args.cred_ivls,
    start_date=start_date,
    end_date=end_date,
)

et = time.time()
print(f"Total time: {et-st:2.3f} s")
