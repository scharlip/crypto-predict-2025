import random
from enum import StrEnum
from typing import List, Tuple

import torch
from torch import nn

from common.common import CoinType, Exchange
from input.input import CoinDataset, MidpointCoinDataset

from tqdm import tqdm

from models.PerfectMidpointPredictorModel import PerfectMidpointPredictorModel
from models.base import BaseModel, TransctionType


def run_backtest(ds: CoinDataset, model: BaseModel, transaction_fee_pctg = 0.006) -> float:

    current_usd_holdings = 100.0
    current_coin_holdings = 0.0
    last_purchased_price = None

    for current_idx in tqdm(range(ds.window_size, len(ds))):
        past_window = ds.df[current_idx - ds.window_size : current_idx]
        current_price = ds.df.iloc[current_idx]["Midpoint"].tolist()

        transaction_info = model.buy_sell_hold_decision(
            past_window,
            last_purchased_price,
            current_usd_holdings > 0.0
        )

        transaction_type = transaction_info[0]
        step_for_transaction = transaction_info[1]

        if transaction_type == TransctionType.Buy and current_idx == step_for_transaction:
            prev_usd_holdings = current_usd_holdings
            prev_coin_holdings = current_coin_holdings

            current_coin_holdings = current_usd_holdings / current_price
            current_coin_holdings *= (1.0 - transaction_fee_pctg)
            current_usd_holdings = 0.0

            last_purchased_price = current_price

            print("Bought coins at a price of {}. USD: {} -> {}, Coin: {} -> {} (Step: {})".format(
                current_price,
                prev_usd_holdings, current_usd_holdings,
                prev_coin_holdings, current_coin_holdings,
                current_idx
            ))
        elif transaction_type == TransctionType.Sell and current_idx == step_for_transaction:
            prev_usd_holdings = current_usd_holdings
            prev_coin_holdings = current_coin_holdings

            current_usd_holdings = current_coin_holdings * current_price
            current_usd_holdings *= (1.0 - transaction_fee_pctg)
            current_coin_holdings = 0.0

            if last_purchased_price < current_price*(1 - transaction_fee_pctg):
                gain_loss_msg = "NET GAIN"
            else:
                gain_loss_msg = "NET LOSS"

            last_purchased_price = None

            print("Sold coins at a price of {}. USD: {} -> {}, Coin: {} -> {} (Step: {}) {}".format(
                current_price,
                prev_usd_holdings, current_usd_holdings,
                prev_coin_holdings, current_coin_holdings,
                current_idx,
                gain_loss_msg
            ))
        else:
            # either hold or not time to buy/sell yet
            pass



ds = MidpointCoinDataset(coin_type=CoinType.ETH, exchange=Exchange.Coinbase)
model = PerfectMidpointPredictorModel(ds, 0.02, 60)

profit = run_backtest(ds, model)