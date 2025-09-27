from pandas import DataFrame

from models.BaseModel import BaseModel
import torch
import os

class MidpointPredictorModel(BaseModel):

    def __init__(self, threshold: float, lookback: int, lookahead: int):
        super().__init__()
        self.threshold = threshold
        self.lookback = lookback
        self.lookahead = lookahead

    def predict_lookahead_window(self, lookback_window: DataFrame) -> DataFrame:
        raise NotImplementedError("Implemented in subclasses")

    def forward(self, x):
        raise NotImplementedError("Implemented in subclasses")

    def save_model(self, model_save_dir: str, **extra_kwargs):
        descriptor = self.descriptor_string()
        filename = "model"
        for (k, v) in extra_kwargs.items():
            filename = filename + "|{}_{}".format(str(k), str(v))

        subdir = model_save_dir + "/" + descriptor
        os.makedirs(subdir, exist_ok=True)
        filename = model_save_dir + "/" + descriptor + "/" + filename + ".pt2"

        torch.save(self, filename)