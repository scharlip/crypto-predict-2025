from pandas import Timestamp

from common.common import CoinType, Exchange
from input.input import CoinDataset
from tqdm.auto import tqdm
from dataclasses import dataclass
import pandas
import os

BASE_DIR = "/Users/scharlip/Desktop/crypto-predict/"
coin_type = CoinType.ETH
exchange = Exchange.Coinbase

ds = CoinDataset(coin_type=coin_type,exchange=exchange)
df = ds.df

df["Midpoint"] = (df["High"] + df["Low"]) / 2

value_thresholds = [0.012, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
window_lengths = [60, 180, 360, 720, 1440]
recency_threshold = pandas.Timestamp.now() - pandas.Timedelta('365D') # 1 year ago

@dataclass
class MatchedWindow:
    difference_percent: float
    start_time: Timestamp
    end_time: Timestamp
    start_midpoint: float
    end_midpoint: float

for window_length in window_lengths:
    for value_threshold in value_thresholds:
        print("Checking {} minute window for {}% rises ...".format(window_length, (value_threshold * 100.0)))

        matches = []
        recent_matches = []

        results_dir = "{}/scratch/investigations/matches/{}_{}/".format(BASE_DIR, str(coin_type), str(exchange))

        if os.path.isdir(results_dir):
            print("Results directory has already been extracted.")
        else:
            os.makedirs(results_dir, exist_ok=True)

        match_filename = "{}/{}_{}.txt".format(results_dir, window_length, value_threshold)
        recent_match_filename = "{}/{}_{}.recent.txt".format(results_dir, window_length, value_threshold)

        with open(match_filename, "w") as match_file, open(recent_match_filename, 'w') as recent_match_file:
            for window in tqdm(df.rolling(window=window_length)):
                if len(window) < window_length:
                    continue

                start_time = window.index.min()
                end_time = window.index.max()

                start_value = window.loc[start_time]["Midpoint"]
                end_value = window.loc[end_time]["Midpoint"]

                difference = end_value - start_value
                difference_percent = difference/start_value

                if difference_percent > value_threshold:
                    matched_window = MatchedWindow(difference_percent, start_time, end_time, start_value, end_value)

                    matches.append(matched_window)
                    match_file.write("{} -> {}: {} -> {} ({})\n".format(
                        matched_window.start_time, matched_window.end_time,
                        matched_window.start_midpoint, matched_window.end_midpoint,
                        matched_window.difference_percent
                    ))
                    if start_time > recency_threshold:
                        recent_matches.append(matched_window)
                        recent_match_file.write("{} -> {}: {} -> {} ({})\n".format(
                            matched_window.start_time, matched_window.end_time,
                            matched_window.start_midpoint, matched_window.end_midpoint,
                            matched_window.difference_percent
                        ))

        print("Done checking {} minute window for {}% rises. Fount {} matches ({} of which were recent).".format(
            window_length, (value_threshold * 100.0), len(matches), len(recent_matches)))
        print()