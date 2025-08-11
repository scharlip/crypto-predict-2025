import atexit

import os
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame

from common.common import CoinType, Exchange, BASE_DIR

WINDOWS_PER_FILE = 20000

def cleanup_temp_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Temporary directory {path} cleaned up on exit.")

class CoinDataset(Dataset):

    def __init__(self, coin_type: CoinType, exchange: Exchange, limit=None,
                 interpolate_missing_data: bool = True, lookback: int = 60, splits=[0.8, 0.1, 0.1]):
        self.coin_type = coin_type
        self.exchange = exchange
        self.limit = limit
        self.interpolate_missing_data = interpolate_missing_data
        self.lookback = lookback
        self.splits = splits
        self.scratch_dir = "{}/scratch/".format(BASE_DIR)
        self.midpoints_dir = "{}/midpoints".format(self.scratch_dir)
        self.df = self.__read_csv(coin_type, exchange, interpolate_missing_data, limit)
        self.cached_midpoints = None

    def __read_csv(self, coin_type: CoinType, exchange: Exchange, interpolate_missing_data: bool = True, limit: int = None) -> DataFrame:
        print("Reading csv for {}/{} ...".format(coin_type, exchange))

        csv = "{}/{}/{}USD_1m_{}.csv".format(BASE_DIR, str(coin_type).lower(), str(coin_type), str(exchange))
        from_csv = pd.read_csv(csv, index_col="Open time")
        from_csv['Exchange'] = str(exchange)
        from_csv['Interpolated'] = False

        extraneous_columns = from_csv.columns.difference(
            ["Open time", "Low", "High", "Open", "Close", "Volume", "Exchange", "Interpolated"])

        if len(extraneous_columns) > 0:
            from_csv.drop(columns=extraneous_columns, axis=1, inplace=True)

        from_csv.index = pd.to_datetime(from_csv.index)

        missing_data = pd.DataFrame(
            {
                "Open time": pd.date_range(
                    start=from_csv.index.min(),
                    end=from_csv.index.max(),
                    freq='min'
                ).difference(from_csv.index)
            }
        )

        missing_data = missing_data.set_index("Open time")
        missing_data[from_csv.columns] = np.nan
        missing_data["Interpolated"] = True
        missing_data["Exchange"] = str(exchange)

        ret = pd.concat([from_csv, missing_data])
        sorted = ret.sort_index()

        if interpolate_missing_data:
            sorted.interpolate(inplace=True, method='linear')

        print("Read csv for {}/{}.".format(coin_type, exchange))

        if limit is not None:
            return sorted.head(limit)
        else:
            return sorted

    def generate_midpoint_windows(self):
        if os.path.isdir(self.midpoints_dir):
            print("Midpoint windows have already been extracted.")
            return

        midpoints = (self.df["High"] + self.df["Low"]) / 2

        dir = "{}/midpoints/{}_{}/".format(self.scratch_dir, str(self.coin_type), str(self.exchange))

        print("Extracting midpoint windows into '{}' ...".format(dir))

        filenum = 0
        count = 0
        os.makedirs(dir, exist_ok=True)
        file = open("{}/{}.txt".format(dir, filenum), "w+")
        for window in tqdm(midpoints.rolling(window=self.lookback + 1)):
            if len(window) < self.lookback + 1:
                continue

            file.write(','.join(map(str, window.tolist())) + "\n")
            count += 1

            if count % WINDOWS_PER_FILE == 0:
                file.flush()
                file.close()

                filenum += 1
                file = open("{}/{}.txt".format(dir, filenum), "w+")

        file.flush()
        file.close()

        print("Done extracting midpoint windows.")
        
    def __len__(self):
        return len(self.df) - self.lookback

    def __getitem__(self, item):
        if item > len(self.df):
            raise IndexError('index out of range')

        file_num = int(item / WINDOWS_PER_FILE)

        if self.cached_midpoints is not None and self.cached_midpoints[0] == file_num:
            cached_windows = self.cached_midpoints[1]
        else:
            with open("{}/{}.txt".format(self.midpoints_dir, file_num)) as f:
                lines = f.readlines()
                cached_windows = [list(map(float, line.strip().split(','))) for line in lines]

            self.cached_midpoints = (file_num, cached_windows)

        offset = int(item % WINDOWS_PER_FILE)

        window = cached_windows[offset]

        return torch.tensor(window[:-1]), torch.tensor(window[-1])


