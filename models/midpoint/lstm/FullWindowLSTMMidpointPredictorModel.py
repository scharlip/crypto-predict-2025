from typing import List

import torch
from pandas import DataFrame
from torch import nn

from models.midpoint.MidpointModelPredictor import MidpointPredictorModel


class FullWindowLSTMMidpointPredictorModel(MidpointPredictorModel):

    def __init__(self, threshold: float, lookback: int, lookahead: int, hidden_size = 50, num_layers = 1, dropout = 0.2, normalizer = None):
        super().__init__(threshold=threshold, lookback=lookback, lookahead=lookahead)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalizer = normalizer
        self.lstm = nn.LSTM(lookback, hidden_size, num_layers, batch_first=True, dropout=dropout).to(self.device)
        self.linear = nn.Linear(hidden_size, lookahead).to(self.device)

    def forward(self, x):
        lstm_hidden, _ = self.lstm(x)
        lstm_hidden = lstm_hidden.to(self.device)
        linear_out = self.linear(lstm_hidden).to(self.device)
        return linear_out

    def predict_lookahead_window(self, lookback_window: DataFrame) -> List[float]:
        if self.normalizer:
            current_window = lookback_window["NormalizedMidpoint"].tolist()
        else:
            current_window = lookback_window["Midpoint"].tolist()

        x_tensor = torch.tensor(current_window).to(self.device)
        x_batch = torch.unsqueeze(x_tensor, 0).to(self.device)
        y_pred = self.forward(x_batch)
        prediction = y_pred[0].tolist()

        future_window = prediction

        if self.normalizer:
            return self.normalizer.denormalize(future_window)
        else:
            return future_window

    def descriptor_string(self):
        format_string = "FullWindowLSTMMidpointPredictorModel_" + \
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
                    self.normalizer is not None
        )
        return descriptor