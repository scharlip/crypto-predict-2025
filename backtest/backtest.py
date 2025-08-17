import random
from enum import StrEnum
from typing import List, Tuple

import torch
from torch import nn

from common.common import CoinType, Exchange
from input.input import CoinDataset, MidpointCoinDataset
from models.models import SimpleLSTMModel

artificial_noise_pctg = 0.01

def predict_future_window(current_idx: int,
                          lookahead: int,
                          past_window: List[float],
                          model: nn.Module,
                          ds: CoinDataset) -> float:

    '''
    # This is the ML approach.

    current_window = past_window.copy()

    for step in range(future_steps):
        pred = model(torch.tensor(current_window))
        current_window.append(pred)
        current_window.pop(0)

    return current_window[-1]
    '''

    perfect_prediction =  ds.df[current_idx + 1 : current_idx + lookahead + 1]["Midpoint"].tolist()

    with_noise = []
    for v in perfect_prediction:
        random.seed(v)
        with_noise.append(v * random.uniform(1.0 - artificial_noise_pctg, 1.0 + artificial_noise_pctg))

    return with_noise

class TransctionType(StrEnum):
    Buy = "Buy"
    Sell = "Sell"
    Hold = "Hold"

def calculate_transaction_info_for_current_step(
        current_idx: int,
        last_purchased_price: float,
        currently_have_usd: bool,
        future_window: List[float],
        threshold: float) -> Tuple[TransctionType, int]:

    min_value = float('inf')
    min_index = -1
    max_value = float('-inf')
    max_index = -1

    for (idx, val) in enumerate(future_window):
        if val > max_value:
            max_value = val
            max_index = idx

        if val < min_value:
            min_value = val
            min_index = idx

    if currently_have_usd:
        # buy at the predicted minimum if the maximum is more than the minimum + the threshold
        if max_value > min_value*(1 + threshold):
            return (TransctionType.Buy, current_idx + min_index)
        else:
            return (TransctionType.Hold, None)
    else:
        # if we predict a value above the threshold, buy there
        if max_value > last_purchased_price*(1 + threshold):
            return (TransctionType.Sell, current_idx + max_index)

        # if we predict a value 3x below the threshold, bail out right now
        elif min_value < last_purchased_price*(1 - 3*threshold):
            return (TransctionType.Sell, current_idx)
        # otherwise do nothing
        else:
            return (TransctionType.Hold, None)

def run_backtest(ds: CoinDataset, model: nn.Module,
                 lookahead = 60, transaction_fee_pctg = 0.006, threshold = 0.02) -> float:

    current_usd_holdings = 100.0
    current_coin_holdings = 0.0
    last_purchased_price = None

    for current_idx in range(ds.window_size, len(ds)):
        past_window = ds.df[current_idx - ds.window_size : current_idx]["Midpoint"].tolist()
        future_window = predict_future_window(current_idx, lookahead, past_window, model, ds)
        current_price = ds.df.iloc[current_idx]["Midpoint"].tolist()

        transaction_info = calculate_transaction_info_for_current_step(
            current_idx, last_purchased_price, current_usd_holdings > 0.0, future_window, threshold)

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
device = torch.device("cpu")
model = SimpleLSTMModel(input_size=ds.window_size, device=device)

profit = run_backtest(ds, model)