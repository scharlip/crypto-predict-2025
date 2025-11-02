from enum import StrEnum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List


class TransctionType(StrEnum):
    Buy = "Buy"
    Sell = "Sell"
    Hold = "Hold"

class TransctionDecisionReason(StrEnum):
    TransactionFeeThreshold = "TransactionFeeThreshold"
    MomentumPriceIncrease = "MomentumPriceIncrease"
    MomentumStalled = "MomentumStalled"
    MomentumPriceDecrease = "MomentumPriceDecrease"
    BailOut = "BailOut"


@dataclass
class TransactionDecision:
    type: TransctionType
    timestamp: datetime
    reason: TransctionDecisionReason = None

class TransactionDecisionHeuristic:

    def decision(self,
        past_window: List[float],
        future_window: List[float],
        current_time: datetime,
        last_purchased_price: float,
        currently_have_usd: bool
    ) -> TransactionDecision:
        raise NotImplementedError("Implemented in subclasses")

class MixedTransactionDecisionHeuristic(TransactionDecisionHeuristic):

    def decision(self,
        past_window: List[float],
        future_window: List[float],
        current_time: datetime,
        last_purchased_price: float,
        currently_have_usd: bool,
        transaction_fee_pctg: float
    ) -> TransactionDecision:
        min_idx = None
        max_idx = None
        min_value = None
        max_value = None

        diffs = []

        for (idx, val) in enumerate(future_window):
            if min_value is None or min_value > val:
                min_value = val
                min_idx = idx

            if max_value is None or max_value < val:
                max_value = val
                max_idx = idx

            if idx > 0:
                diffs.append(future_window[idx] - future_window[idx - 1])

        min_datetime = current_time + timedelta(minutes=min_idx)
        max_datetime = current_time + timedelta(minutes=max_idx)

        up_or_down = [v > 0 for v in diffs]
        num_up = sum(up_or_down)
        num_down = len(up_or_down) - num_up

        if currently_have_usd:
            # buy at the predicted minimum if the maximum is more than the minimum + the transaction_fee_pctg
            if max_value > min_value * (1 + transaction_fee_pctg) and max_idx > min_idx:
                return TransactionDecision(TransctionType.Buy, min_datetime, TransctionDecisionReason.TransactionFeeThreshold)
            elif num_up > 0.9 * len(up_or_down) and future_window[0] < future_window[-1]:
                return TransactionDecision(TransctionType.Buy, current_time, TransctionDecisionReason.MomentumPriceIncrease)
            else:
                return TransactionDecision(TransctionType.Hold, None)
        else:
            if num_down > 0.9 * len(up_or_down) and future_window[0] > future_window[-1]:
                return TransactionDecision(TransctionType.Sell, current_time, TransctionDecisionReason.MomentumPriceDecrease)
            if num_down > 0.3 * len(up_or_down) and num_down < 0.6 * len(up_or_down):
                return TransactionDecision(TransctionType.Sell, current_time, TransctionDecisionReason.MomentumStalled)
            # if we predict a value above the threshold, sell there
            elif max_value > last_purchased_price * (1 + transaction_fee_pctg):
                return TransactionDecision(TransctionType.Sell, max_datetime, TransctionDecisionReason.TransactionFeeThreshold)
            # if we predict a value 3x below the threshold, bail out right now
            elif min_value < last_purchased_price * (1 - 3 * transaction_fee_pctg):
                return TransactionDecision(TransctionType.Sell, current_time, TransctionDecisionReason.BailOut)
            # otherwise do nothing
            else:
                return TransactionDecision(TransctionType.Hold, None)