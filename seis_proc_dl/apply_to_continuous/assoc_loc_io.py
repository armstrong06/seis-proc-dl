import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from seis_proc_db import services
from seis_proc_db.database import Session


def make_input_files():
    p_repicker_method = "P-MSWAG-Armstrong2023"
    s_repicker_method = "S-MSWAG-Armstrong2023"
    p_calibration_method = "P-Kuleshov2018-Armstrong2023"
    s_calibration_method = "S-Kuleshov2018-Armstrong2023"
    start_date = "2023-01-01"
    end_date = "2023-01-02"
    p_max_width = 0.30
    s_max_width = 0.40
    p_min_width = 0.150
    s_min_width = 0.250
    ci_perc = 90
    outdir = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/process_ys_data/assoc_loc_io/in"

    with Session() as session:
        p_pick_df = services.make_pick_catalog_df(
            session,
            "P",
            p_repicker_method,
            p_calibration_method,
            ci_perc,
            start=start_date,
            end=end_date,
            max_width=p_max_width,
            min_width=p_min_width,
        )
        s_pick_df = services.make_pick_catalog_df(
            session,
            "S",
            s_repicker_method,
            s_calibration_method,
            ci_perc,
            start=start_date,
            end=end_date,
            max_width=s_max_width,
            min_width=s_min_width,
        )

    pick_df = pd.concat([p_pick_df, s_pick_df]).sort_values("arrival_time").round({"uncertainty": 3})

    dateformat = "%Y-%m-%d"
    curr_date = datetime.strptime(start_date, dateformat)
    last_date = datetime.strptime(end_date, dateformat)
    delta = timedelta(days = 1)

    while curr_date < last_date:
        curr_day_epoch = curr_date.replace(tzinfo=timezone.utc).timestamp()
        next_day_epoch = (curr_date + delta).replace(tzinfo=timezone.utc).timestamp()
        day_df = pick_df[(pick_df["arrival_time"] >= curr_day_epoch) & (pick_df["arrival_time"] < next_day_epoch)]
        day_df.to_csv(os.path.join(outdir, f"{curr_date.strftime(dateformat)}.csv"), index=False)

        curr_date += delta

if __name__ == "__main__":
    make_input_files()
