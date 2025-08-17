from datetime import datetime
from enum import StrEnum
from typing import List, Tuple

import torch.nn as nn
from pandas import DataFrame


class TransctionType(StrEnum):
    Buy = "Buy"
    Sell = "Sell"
    Hold = "Hold"

class BaseModel(nn.Module):

    def buy_sell_hold_decision(
            self,
            current_time: datetime,
            past_window: DataFrame,
            last_purchased_price: float,
            currently_have_usd: bool,
    ) -> Tuple[TransctionType, int]:
        raise NotImplementedError("Implemented in subclasses")

    def forward(self, x):
        raise NotImplementedError("Implemented in subclasses")