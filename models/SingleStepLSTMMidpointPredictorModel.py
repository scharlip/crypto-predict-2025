from datetime import datetime
from typing import Tuple, List

import torch
from pandas import DataFrame
from torch import nn

from models.MidpointModelPredictor import MidpointPredictorModel
from models.BaseModel import TransctionType


class SingleStepLSTMMidpointPredictorModel(MidpointPredictorModel):

    def __init__(self, threshold: float, lookback: int, lookahead: int, hidden_size = 50, num_layers = 1, dropout = 0.2, is_data_normalized = False):
        super().__init__(threshold=threshold, lookback=lookback, lookahead=lookahead)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.is_data_normalized = is_data_normalized
        self.lstm = nn.LSTM(lookback, hidden_size, num_layers, batch_first=True, dropout=dropout).to(self.device)
        self.linear = nn.Linear(hidden_size, 1).to(self.device)

    def forward(self, x):
        lstm_hidden, _ = self.lstm(x)
        lstm_hidden = lstm_hidden.to(self.device)
        linear_out = self.linear(lstm_hidden).to(self.device)
        return linear_out

    def predict_lookahead_window(self, lookback_window: DataFrame) -> List[float]:
        if self.is_data_normalized:
            current_window = lookback_window["NormalizedMidpoint"].tolist()
        else:
            current_window = lookback_window["Midpoint"].tolist()

        future_window = []

        while len(future_window) < self.lookahead:
            x_tensor = torch.tensor(current_window).to(self.device)
            x_batch = torch.unsqueeze(x_tensor, 0).to(self.device)
            y_pred = self.forward(x_batch)
            prediction = y_pred[0][0].tolist()
            future_window.append(prediction)
            current_window.append(prediction)
            current_window.pop(0)

        if self.is_data_normalized:
            return [(10.0**v) for v in future_window]
        else:
            return future_window

        return future_window

    def descriptor_string(self):
        format_string = "SingleStepLSTMMidpointPredictorModel_" + \
                "lookback_{}_" + \
                "lookahead_{}_" + \
                "hidden_{}_" + \
                "layers_{}_" + \
                "dropout_{}_" + \
                "normalized_{}"
        descriptor = format_string.format(
                    self.lookback,
                    self.lookahead,
                    self.hidden_size,
                    self.num_layers,
                    self.dropout,
                    self.is_data_normalized
                )
        return descriptor