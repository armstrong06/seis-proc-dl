import obspy
import os
import numpy as np
import argparse
from datetime import datetime
from seis_proc_db.database import Session
from seis_proc_db.services import get_operating_channels

argParser = argparse.ArgumentParser()
argParser.add_argument(
    "--ddir",
    type=str,
    help="path to data directory",
    default="/uufs/chpc.utah.edu/common/home/koper-group3/alysha/ys_data/downloaded_all_data",
)
argParser.add_argument(
    "--min_date", type=str, help="Beginning of range to gather stations for"
)
argParser.add_argument(
    "--max_date", type=str, help="End of range to gather stations for"
)
# argParser.add_argument("-c", "--ncomps", type=int, help="number of components")
# argParser.add_argument(
#     "-s",
#     "--stattype",
#     type=str,
#     help="first letter(s) of the channel code",
#     default="HEB",
# )
argParser.add_argument("--outdir", type=str, help="output directory", default=None)
# argParser.add_argument(
#     "-f", "--outfile", type=str, help="output filename", default=None
# )
argParser.add_argument(
    "-n", "--nstats", type=int, help="limit on the number of stations", default=None
)
args = argParser.parse_args()

min_date = args.min_date
max_date = args.max_date
# ncomps = args.ncomps
# stat_type = args.stattype
outdir = args.outdir
# outfile = args.outfile
ddir = args.ddir
nstats = args.nstats

# assert ncomps in [1, 3], "Invalid number of components"

dateformat = "%Y-%m-%d"
min_date = datetime.strptime(min_date, dateformat)
max_date = datetime.strptime(max_date, dateformat)

with Session() as session:
    with session.begin():
        channel_infos = get_operating_channels(session, min_date, max_date)

chan_list = [f"{ci[0]}.{ci[1]}.{ci[2]}.{ci[3][0:2]}" for ci in channel_infos]

uniq_chans, chan_cnts = np.unique(chan_list, return_counts=True)

threec_chans = uniq_chans[np.where((chan_cnts >= 3) & (chan_cnts % 3 == 0))[0]]
onec_chans = uniq_chans[np.where(chan_cnts < 3)[0]]

if nstats is not None:
    threec_chans = threec_chans[0:nstats]
    onec_chans = onec_chans[0:nstats]

for ncomps, selected_chans in zip((1, 3), (onec_chans, threec_chans)):
    outfile = f"station.list.db.{ncomps}C.{min_date.date()}.{max_date.date()}"
    if nstats is not None:
        outfile == f"{nstats}stats.txt"

    outfile += ".txt"

    if outdir is not None:
        outfile = os.path.join(outdir, outfile)

    with open(os.path.join(outfile), "w") as f:
        f.write(f"{len(selected_chans)}\n")
        for chan in selected_chans:
            chan_comps = chan.split(".")
            f.write(
                f'{chan_comps[0]} {chan_comps[1]} "{chan_comps[2]}" {chan_comps[3]}\n'
            )
