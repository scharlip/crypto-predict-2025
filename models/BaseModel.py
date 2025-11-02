import torch.nn as nn
from pandas import DataFrame


class BaseModel(nn.Module):

    def predict_lookahead_window(self, lookback_window: DataFrame) -> DataFrame:
        raise NotImplementedError("Implemented in subclasses")

    def descriptor_string(self):
        raise NotImplementedError("Implemented in subclasses")