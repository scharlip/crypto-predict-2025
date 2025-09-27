import random
import math
from datetime import timedelta

from tqdm.auto import tqdm

from input.coindataset import CoinDataset
from models.BaseModel import BaseModel, TransctionType
from models.midpoint.MidpointModelPredictor import MidpointPredictorModel

def log(msg: str, log_file):
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()
    else:
        print(msg + "\n")

def run_backtest(ds: CoinDataset, model: BaseModel, transaction_fee_pctg = 0.006, log_file_name = None, should_log = True) -> float:

    log_file = None
    if log_file_name:
        log_file = open(log_file_name, 'w')

    if should_log:
        log("Starting backtest ...", log_file)

    total_transactions = 0
    held_for = []

    current_usd_holdings = 100.0
    current_coin_holdings = 0.0

    last_purchased_price = None
    last_purchased_time = None

    for current_idx in tqdm(range(ds.lookback, len(ds))):
        past_window = ds.df[current_idx - ds.lookback: current_idx]
        current_price = ds.df.iloc[current_idx]["Midpoint"].tolist()
        current_time = ds.df.iloc[current_idx]["Open time"].to_pydatetime()

        transaction_info = model.buy_sell_hold_decision(
            current_time,
            past_window,
            last_purchased_price,
            current_usd_holdings > 0.0
        )

        transaction_type = transaction_info[0]
        timestamp_for_transaction = transaction_info[1]

        if transaction_type == TransctionType.Buy and timestamp_for_transaction <= current_time:
            prev_usd_holdings = current_usd_holdings
            prev_coin_holdings = current_coin_holdings

            current_coin_holdings = current_usd_holdings / current_price
            current_coin_holdings *= (1.0 - transaction_fee_pctg)
            current_usd_holdings = 0.0

            if should_log:
                log("\nBought coins at a price of {}. USD: {} -> {}, Coin: {} -> {} (Step: {}, Total time elapsed: {})".format(
                    current_price,
                    prev_usd_holdings, current_usd_holdings,
                    prev_coin_holdings, current_coin_holdings,
                    current_idx,
                    str(timedelta(minutes=current_idx)),
                ), log_file)

            last_purchased_price = current_price
            last_purchased_time = current_time
            total_transactions += 1
        elif transaction_type == TransctionType.Sell and timestamp_for_transaction <= current_time:
            prev_usd_holdings = current_usd_holdings
            prev_coin_holdings = current_coin_holdings

            current_usd_holdings = current_coin_holdings * current_price
            current_usd_holdings *= (1.0 - transaction_fee_pctg)
            current_coin_holdings = 0.0

            if should_log:
                if last_purchased_price < current_price * (1 - 2*transaction_fee_pctg):
                    gain_loss_msg = "NET GAIN"
                else:
                    gain_loss_msg = "NET LOSS"

                log("\nSold coins at a price of {}. USD: {} -> {}, Coin: {} -> {} (Held for: {}, Total time elapsed: {}, Step: {}) {}".format(
                    current_price,
                    prev_usd_holdings, current_usd_holdings,
                    prev_coin_holdings, current_coin_holdings,
                    str(current_time - last_purchased_time),
                    str(timedelta(minutes = current_idx)),
                    current_idx,
                    gain_loss_msg
                ), log_file)

            held_for.append(current_time - last_purchased_time)
            last_purchased_price = None
            last_purchased_time = None
            total_transactions += 1
        else:
            # either hold or not time to buy/sell yet
            pass

    if len(held_for) > 0:
        average_hold_time = sum(held_for, timedelta(0))/len(held_for)
    else:
        average_hold_time = "N/A"

    if should_log:
        log("Backtest completed. Final results:", log_file)
        log("  USD: {}".format(current_usd_holdings), log_file)
        log("  Coin: {}".format(current_coin_holdings), log_file)
        log("  Final coin price: {}".format(current_price), log_file)
        log("  Total Transactions: {}".format(total_transactions), log_file)
        log("  Average Hold Time: {}".format(str(average_hold_time)), log_file)
        log("  Time elapsed: {}".format(str(timedelta(minutes = current_idx))), log_file)
        log("  Steps: {}".format(current_idx), log_file)

    if current_usd_holdings == 0.0:
        current_usd_holdings = current_coin_holdings * current_price
        current_usd_holdings *= (1.0 - transaction_fee_pctg)

    if log_file:
        log_file.flush()
        log_file.close()

    return current_usd_holdings

def spot_check(ds: CoinDataset, model: MidpointPredictorModel, num_spot_checks = 20, seed: int = 42, log_file_name = None, should_log = True):
    log_file = None
    if log_file_name:
        log_file = open(log_file_name, 'w')

    random.seed(seed)
    all_absolute_diffs = []
    all_pctg_diffs = []
    for _ in range(num_spot_checks):
        idx = random.randint(model.lookahead, len(ds.df))
        past_window = ds.df[idx - model.lookback : idx]
        future_window = ds.df[idx : idx + model.lookahead]
        future_vals = future_window["Midpoint"].tolist()
        predicted_vals = model.predict_lookahead_window(past_window)

        absolute_diffs = [math.fabs(predicted_vals[i] - future_vals[i]) for i in range(len(predicted_vals))]
        pctg_diffs = [absolute_diffs[i]/future_vals[i] for i in range(len(future_vals))]

        all_absolute_diffs.extend(absolute_diffs)
        all_pctg_diffs.extend(pctg_diffs)

    avg_absolute_diff = sum(all_absolute_diffs)/len(all_absolute_diffs)
    avg_pctg_diff = sum(all_pctg_diffs)/len(all_pctg_diffs)

    max_absolute_diff = max(all_absolute_diffs)
    max_pctg_diff = max(all_pctg_diffs)

    min_absolute_diff = min(all_absolute_diffs)
    min_pctg_diff = min(all_pctg_diffs)

    if should_log:
        log("Spot check results: {} spot checks (seed={})".format(num_spot_checks, seed), log_file)
        log("  Averages: absolute={} percentage={}".format(avg_absolute_diff, avg_pctg_diff), log_file)
        log("  Minimums: absolute={} percentage={}".format(min_absolute_diff, min_pctg_diff), log_file)
        log("  Maximums: absolute={} percentage={}".format(max_absolute_diff, max_pctg_diff), log_file)

    if log_file:
        log_file.flush()
        log_file.close()