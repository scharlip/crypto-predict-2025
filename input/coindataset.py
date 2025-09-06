from datetime import datetime
from typing import Tuple

import numpy as np
import torch
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
                 rolling_stats_len = 24*60,
                 use_normalized_data = False
                 ):
        self.csv_dir = csv_dir
        self.coin_type = coin_type
        self.exchange = exchange
        self.limit = limit
        self.date_range_filter = date_range_filter
        self.interpolate_missing_data = interpolate_missing_data
        self.window_size = window_size
        self.rolling_stats_len = rolling_stats_len
        self.use_normalized_data = use_normalized_data
        self.df = self.__read_csv()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def __read_csv(self) -> DataFrame:
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

        if self.interpolate_missing_data:
            ret.interpolate(inplace=True, method='linear')

        ret["Midpoint"] = (ret["Low"] + ret["High"])/2
        ret["RollingMean"] = ret["Midpoint"].rolling(self.rolling_stats_len).mean()
        ret["RollingStdDev"] = ret["Midpoint"].rolling(self.rolling_stats_len).std()
        ret["NormalizedMidpoint"] = (ret["Midpoint"] - ret["RollingMean"])/ret["RollingStdDev"]

        print("Read csv for {}/{}.".format(self.coin_type, self.exchange))

        if self.date_range_filter is not None:
            start = pd.Timestamp(self.date_range_filter[0])
            end = pd.Timestamp(self.date_range_filter[1])
            ret = ret[(ret["Open time"] >= start) & (ret["Open time"] <= end)]
            ret = ret.reset_index()

        if self.limit is not None:
            return ret.head(self.limit)
        else:
            return ret