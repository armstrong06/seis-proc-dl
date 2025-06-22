import calendar
from datetime import date, timedelta
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-y", "--year", type=int, help="year to gather stations")
argParser.add_argument("-s", "--start", type=int, help="starting month", default=1)
argParser.add_argument("-e", "--end", type=int, help="ending month", default=12)

args = argParser.parse_args()
year = args.year
start_month = args.start 
end_month = args.end

assert year > 2001 and year < 2025, "Invalid year"
assert start_month > 0 and start_month < 13, "Start month should be between 1 and 12"
assert end_month > 0 and end_month < 13, "End month should be between 1 and 12"
assert start_month <= end_month, "End month should be greater than or equal to start month"

outfile = f"./rundates.{year}.{start_month:02d}.{end_month:02d}.txt"

start_end_dates = []
for month in range(start_month, end_month + 1):
    # Get the last day of the month
    last_day = calendar.monthrange(year, month)[1]
    start_date = date(year, month, 1)
    end_date = date(year, month, last_day) + timedelta(days=1)
    start_end_dates.append((start_date, end_date))

with open(outfile, "w") as f:
    f.write(f"{len(start_end_dates)}\n")
    for month in start_end_dates:
        f.write(f'{month[0]} {month[1]}\n')
