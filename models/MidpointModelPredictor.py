from pandas import DataFrame

from models.BaseModel import BaseModel, TransctionType

class MidpointPredictorModel(BaseModel):

    def __init__(self, threshold: float, lookahead: int):
        super().__init__()
        self.threshold = threshold
        self.lookahead = lookahead
        self.cache = None

    def predict_future_window(self, past_window: DataFrame) -> DataFrame:
        raise NotImplementedError("Implemented in subclasses")

    def forward(self, x):
        raise NotImplementedError("Implemented in subclasses")