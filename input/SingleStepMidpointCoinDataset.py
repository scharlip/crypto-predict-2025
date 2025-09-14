from datetime import datetime
from typing import Tuple

import torch
from torch.utils.data import Dataset

from common.common import CoinType, Exchange
from input.coindataset import CoinDataset


class SingleStepMidpointCoinDataset(CoinDataset, Dataset):

    def __init__(self,
                 csv_dir: str,
                 coin_type: CoinType,
                 exchange: Exchange,
                 lookback: int,
                 lookahead: int,
                 limit: int = None,
                 date_range_filter: Tuple[datetime, datetime] = None,
                 interpolate_missing_data: bool = True,
                 use_normalized_data = False):
        super().__init__(csv_dir=csv_dir, coin_type=coin_type, exchange=exchange,
                         lookback=lookback, lookahead=lookahead,
                         limit=limit, date_range_filter=date_range_filter,
                         interpolate_missing_data=interpolate_missing_data,
                         use_normalized_data=use_normalized_data)

    def __len__(self):
        return len(self.df) - self.lookback - 1

    def __getitem__(self, item):
        if item > len(self.df):
            raise IndexError('index out of range')

        if self.use_normalized_data:
            X = self.df[item : item + self.lookback]["NormalizedMidpoint"].tolist()
            y = self.df.iloc[item + self.lookback + 1]["NormalizedMidpoint"].tolist()
        else:
            X = self.df[item : item + self.lookback]["Midpoint"].tolist()
            y = self.df.iloc[item + self.lookback + 1]["Midpoint"].tolist()

        return torch.tensor(X).to(self.device), torch.tensor(y).to(self.device)