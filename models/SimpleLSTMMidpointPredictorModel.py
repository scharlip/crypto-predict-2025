from datetime import datetime
from typing import Tuple, List

import torch
from pandas import DataFrame
from torch import nn

from models.MidpointModelPredictor import MidpointPredictorModel
from models.BaseModel import TransctionType


class SimpleLSTMMidpointPredictorModel(MidpointPredictorModel):

    def __init__(self, threshold: float, lookahead: int, hidden_size = 50, num_layers = 1, dropout = 0.2):
        super().__init__(threshold, lookahead)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(lookahead, hidden_size, num_layers, batch_first=True, dropout=dropout).to(self.device)
        self.linear = nn.Linear(hidden_size, 1).to(self.device)

    def forward(self, x):
        lstm_hidden, _ = self.lstm(x)
        lstm_hidden = lstm_hidden.to(self.device)
        linear_out = self.linear(lstm_hidden).to(self.device)
        return linear_out

    def predict_future_window(self, past_window: DataFrame) -> List[float]:
        current_window = past_window["Midpoint"].tolist()
        future_window = []

        while len(future_window) < self.lookahead:
            x_tensor = torch.tensor(current_window).to(self.device)
            x_batch = torch.unsqueeze(x_tensor, 0).to(self.device)
            y_pred = self.forward(x_batch)
            prediction = y_pred[0][0].tolist()
            future_window.append(prediction)
            current_window.append(prediction)
            current_window.pop(0)

        return future_window

    def buy_sell_hold_decision(
            self,
            current_time: datetime,
            past_window: DataFrame,
            last_purchased_price: float,
            currently_have_usd: bool) -> Tuple[TransctionType, int]:

        future_window = self.predict_future_window(past_window)
        return super().decision(current_time, future_window, last_purchased_price, currently_have_usd)

