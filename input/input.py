from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame

from common.common import CoinType, Exchange

class CoinDataset:

    def __init__(self,
                 csv_dir: str,
                 coin_type: CoinType,
                 exchange: Exchange,
                 limit: int = None,
                 date_range_filter: Tuple[datetime, datetime] = None,
                 interpolate_missing_data: bool = True,
                 window_size: int = 60,
                 ):
        self.csv_dir = csv_dir
        self.coin_type = coin_type
        self.exchange = exchange
        self.limit = limit
        self.date_range_filter = date_range_filter
        self.interpolate_missing_data = interpolate_missing_data
        self.window_size = window_size
        self.df = self.__read_csv(interpolate_missing_data, limit, date_range_filter)

    def __read_csv(self,  interpolate_missing_data: bool = True, limit: int = None, date_range_filter: Tuple[datetime, datetime] = None) -> DataFrame:
        print("Reading csv for {}/{} (at {}) ...".format(self.coin_type, self.exchange, self.csv_dir))

        csv = "{}/{}/{}USD_1m_{}.csv".format(self.csv_dir, str(self.coin_type).lower(), str(self.coin_type), str(self.exchange))
        from_csv = pd.read_csv(csv)
        from_csv['Exchange'] = str(self.exchange)
        from_csv['Interpolated'] = False

        extraneous_columns = from_csv.columns.difference(
            ["Open time", "Low", "High", "Open", "Close", "Volume", "Exchange", "Interpolated"])

        if len(extraneous_columns) > 0:
            from_csv.drop(columns=extraneous_columns, axis=1, inplace=True)

        from_csv["Open time"] = pd.to_datetime(from_csv["Open time"])

        missing_data = pd.DataFrame(
            {
                "Open time": pd.date_range(
                    start=from_csv["Open time"].min(),
                    end=from_csv["Open time"].max(),
                    freq='min'
                ).difference(from_csv["Open time"])
            }
        )

        missing_data[from_csv.columns.difference(["Open time"])] = np.nan
        missing_data["Interpolated"] = True
        missing_data["Exchange"] = str(self.exchange)

        ret = pd.concat([from_csv, missing_data])

        ret = ret.sort_values("Open time").reset_index(drop=True)

        if interpolate_missing_data:
            ret.interpolate(inplace=True, method='linear')

        ret["Midpoint"] = (ret["Low"] + ret["High"])/2

        print("Read csv for {}/{}.".format(self.coin_type, self.exchange))

        if date_range_filter is not None:
            start = pd.Timestamp(date_range_filter[0])
            end = pd.Timestamp(date_range_filter[1])
            ret = ret[(ret["Open time"] >= start) & (ret["Open time"] <= end)]
            ret = ret.reset_index()

        if limit is not None:
            return ret.head(limit)
        else:
            return ret

class MidpointCoinDataset(CoinDataset, Dataset):

    def __init__(self,
                 csv_dir: str,
                 coin_type: CoinType,
                 exchange: Exchange,
                 limit: int = None,
                 date_range_filter: Tuple[datetime, datetime] = None,
                 interpolate_missing_data: bool = True,
                 window_size: int = 60):
        super().__init__(csv_dir, coin_type, exchange, limit, date_range_filter, interpolate_missing_data, window_size)

    def __len__(self):
        return len(self.df) - self.window_size - 1

    def __getitem__(self, item):
        if item > len(self.df):
            raise IndexError('index out of range')

        X = self.df[item : item + self.window_size]["Midpoint"].tolist()
        y = self.df.iloc[item + self.window_size + 1]["Midpoint"].tolist()

        return torch.tensor(X), torch.tensor(y)
