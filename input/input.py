import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame
from dataclasses import dataclass

from common.common import CoinType, Exchange

BASE_DIR = "/Users/scharlip/Desktop/crypto-predict/data/"

def read_csv(coin_type: CoinType, exchange: Exchange, interpolate_missing_data: bool = True) -> DataFrame:
    csv = "{}/{}/{}USD_1m_{}.csv".format(BASE_DIR, str(coin_type).lower(), str(coin_type), str(exchange))
    from_csv = pd.read_csv(csv, index_col="Open time")
    from_csv['Exchange'] = str(exchange)
    from_csv['Interpolated'] = False

    extraneous_columns = from_csv.columns.difference(["Open time", "Low", "High", "Open", "Close", "Volume", "Exchange", "Interpolated"])

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

    return sorted

def read_combined_csv(coin_type: CoinType) -> DataFrame:
    combined_df = None

    for exchage in tqdm(Exchange):
        df_for_exchange = read_csv(CoinType.ETH, exchage)

        if combined_df is None:
            combined_df = df_for_exchange
        else:
            combined_df = pd.concat([combined_df, df_for_exchange])

    return combined_df.sort_index()

@dataclass
class DataSet:
    X_train: torch.tensor
    y_train: torch.tensor

    X_validate: torch.tensor
    y_validate: torch.tensor

    X_test: torch.tensor
    y_test: torch.tensor

def construct_midpoint_dataset(df: DataFrame, lookback: int, splits = [0.8, 0.1, 0.1]) -> DataSet:
    df["Midpoint"] = (df["High"] + df["Low"])/2
    dataset = df[["Midpoint"]].values.astype('float32')

    train_size = int(len(dataset) * splits[0])
    validate_size = int(len(dataset) * splits[1])
    test_size = int(len(dataset) * splits[2])

    train_range = range(0, train_size)
    validate_range = range(train_size, train_size + validate_size)
    test_range = range(train_size + validate_size, len(dataset))


    X_train, y_train = [], []
    for i in range(0, train_size - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X_train.append(feature)
        y_train.append(target)

    X_validate, y_validate = [], []
    for i in range(train_size - lookback, train_size + validate_size - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X_validate.append(feature)
        y_validate.append(target)

    X_test, y_test = [], []
    for i in range(train_size + validate_size - lookback, len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X_test.append(feature)
        y_test.append(target)

    return DataSet(
        X_train=torch.tensor(X_train),
        y_train=torch.tensor(y_train),
        X_validate=torch.tensor(X_validate),
        y_validate=torch.tensor(y_validate),
        X_test=torch.tensor(X_test),
        y_test=torch.tensor(y_test),
    )