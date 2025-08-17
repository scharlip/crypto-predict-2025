from enum import StrEnum
from typing import List, Tuple

import torch.nn as nn
from pandas import DataFrame


class TransctionType(StrEnum):
    Buy = "Buy"
    Sell = "Sell"
    Hold = "Hold"

class BaseModel(nn.Module):

    #def predict_future_window(self, current_idx: int, past_window: List[float], lookahead: int) -> List[int]:
    #    raise NotImplementedError()

    def buy_sell_hold_decision(
            self,
            past_window: DataFrame,
            last_purchased_price: float,
            currently_have_usd: bool,
    ) -> Tuple[TransctionType, int]:
        raise NotImplementedError()