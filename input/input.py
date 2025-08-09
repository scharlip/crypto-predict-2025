import numpy as np
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame, Index
from numpy import arange
from enum import StrEnum

BASE_DIR = "/Users/scharlip/Desktop/crypto-predict/data/"

class CoinType(StrEnum):
    BTC = "BTC"
    ETH = "ETH"
    ADA = "ADA"

class Exchange(StrEnum):
    Binance = "Binance"
    Bitfinex = "Bitfinex"
    BitMEX = "BitMEX"
    Bitstamp = "Bitstamp"
    Coinbase = "Coinbase"
    KuCoin = "KuCoin"

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

