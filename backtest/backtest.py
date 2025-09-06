from datetime import timedelta

from input.input import CoinDataset

from tqdm.auto import tqdm

from models.BaseModel import BaseModel, TransctionType


def run_backtest(ds: CoinDataset, model: BaseModel, transaction_fee_pctg = 0.006, print_debug_statements = False) -> float:
    if print_debug_statements:
        print("Starting backtest ...")

    total_transactions = 0
    held_for = []

    current_usd_holdings = 100.0
    current_coin_holdings = 0.0

    last_purchased_price = None
    last_purchased_time = None

    for current_idx in tqdm(range(ds.window_size, len(ds))):
        past_window = ds.df[current_idx - ds.window_size : current_idx]
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

            if print_debug_statements:
                print("\nBought coins at a price of {}. USD: {} -> {}, Coin: {} -> {} (Step: {}, Total time elapsed: {})".format(
                    current_price,
                    prev_usd_holdings, current_usd_holdings,
                    prev_coin_holdings, current_coin_holdings,
                    current_idx,
                    str(timedelta(minutes=current_idx)),
                ))

            last_purchased_price = current_price
            last_purchased_time = current_time
            total_transactions += 1
        elif transaction_type == TransctionType.Sell and timestamp_for_transaction <= current_time:
            prev_usd_holdings = current_usd_holdings
            prev_coin_holdings = current_coin_holdings

            current_usd_holdings = current_coin_holdings * current_price
            current_usd_holdings *= (1.0 - transaction_fee_pctg)
            current_coin_holdings = 0.0

            if print_debug_statements:
                if last_purchased_price < current_price * (1 - 2*transaction_fee_pctg):
                    gain_loss_msg = "NET GAIN"
                else:
                    gain_loss_msg = "NET LOSS"

                print("\nSold coins at a price of {}. USD: {} -> {}, Coin: {} -> {} (Held for: {}, Total time elapsed: {}, Step: {}) {}".format(
                    current_price,
                    prev_usd_holdings, current_usd_holdings,
                    prev_coin_holdings, current_coin_holdings,
                    str(current_time - last_purchased_time),
                    str(timedelta(minutes = current_idx)),
                    current_idx,
                    gain_loss_msg
                ))

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

    if print_debug_statements:
        print("Backtest completed. Final results:")
        print("  USD: {}".format(current_usd_holdings))
        print("  Coin: {}".format(current_coin_holdings))
        print("  Final coin price: {}".format(current_price))
        print("  Total Transactions: {}".format(total_transactions))
        print("  Average Hold Time: {}".format(str(average_hold_time)))
        print("  Time elapsed: {}".format(str(timedelta(minutes = current_idx))))
        print("  Steps: {}".format(current_idx))

    if current_usd_holdings == 0.0:
        current_usd_holdings = current_coin_holdings * current_price
        current_usd_holdings *= (1.0 - transaction_fee_pctg)

    return current_usd_holdings
