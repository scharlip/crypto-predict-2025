import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame

from common.common import CoinType, Exchange, BASE_DIR

class CoinDataset(Dataset):

    def __init__(self, coin_type: CoinType, exchange: Exchange,
                 limit=None,
                 interpolate_missing_data: bool = True,
                 lookback_window_size: int = 60,
                 ):
        self.coin_type = coin_type
        self.exchange = exchange
        self.limit = limit
        self.interpolate_missing_data = interpolate_missing_data
        self.lookback_window_size = lookback_window_size
        self.df = self.__read_csv(coin_type, exchange, interpolate_missing_data, limit)

    def __read_csv(self, coin_type: CoinType, exchange: Exchange, interpolate_missing_data: bool = True, limit: int = None) -> DataFrame:
        print("Reading csv for {}/{} ...".format(coin_type, exchange))

        csv = "{}/{}/{}USD_1m_{}.csv".format(BASE_DIR, str(coin_type).lower(), str(coin_type), str(exchange))
        from_csv = pd.read_csv(csv)
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
        ret.sort_values("Open time", inplace=True)

        if interpolate_missing_data:
            ret.interpolate(inplace=True, method='linear')

        ret["Midpoint"] = (ret["Low"] + ret["High"])/2

        print("Read csv for {}/{}.".format(coin_type, exchange))

        if limit is not None:
            return ret.head(limit)
        else:
            return ret
        
    def __len__(self):
        return len(self.df) - self.lookback_window_size - 1

    def __getitem__(self, item):
        if item > len(self.df):
            raise IndexError('index out of range')

        X = self.df[item : item + self.lookback_window_size]["Midpoint"].tolist()
        y = self.df.iloc[item + self.lookback_window_size + 1]["Midpoint"].tolist()

        return torch.tensor(X), torch.tensor(y)
