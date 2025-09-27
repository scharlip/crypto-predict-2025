from typing import List

from pandas import DataFrame

from input.SingleStepMidpointCoinDataset import SingleStepMidpointCoinDataset
import random

from models.midpoint.MidpointModelPredictor import MidpointPredictorModel


class NoisySourceMidpointPredictorModel(MidpointPredictorModel):

    def __init__(self, ds: SingleStepMidpointCoinDataset, threshold: float, lookback: int, lookahead: int, artificial_noise_pctg: float = 0.01) -> List[float]:
        super().__init__(threshold, lookback=lookback, lookahead=lookahead)
        self.df = ds.df
        self.artificial_noise_pctg = artificial_noise_pctg
        self.add_noise_function = self.__add_noise_function(artificial_noise_pctg)

    def __add_noise_function(self, artificial_noise_pctg: float) :
        def __noisify(val: float):
            random.seed(val)
            return val * random.uniform(1.0 - artificial_noise_pctg, 1.0 + artificial_noise_pctg)
        return __noisify

    def predict_lookahead_window(self, lookback_window: DataFrame) -> List[float]:
        end_timestamp = lookback_window.iloc[-1]["Open time"]
        end_index = self.df[self.df["Open time"] == end_timestamp].index.tolist()[0]
        perfect_prediction_window = self.df[end_index + 1: end_index + self.lookahead + 1].copy()
        perfect_prediction_window["Midpoint"] = perfect_prediction_window["Midpoint"].apply(self.add_noise_function)
        return perfect_prediction_window["Midpoint"].tolist()

    def descriptor_string(self):
        format_string = "NoisySourceMidpointPredictorModel_" + \
                "threshold_{}_" + \
                "lookback_{}_" + \
                "lookahead_{}_" + \
                "artificial_noise_pctg_{}"
        descriptor = format_string.format(
                    self.threshold,
                    self.lookback,
                    self.lookahead,
                    self.artificial_noise_pctg
                )
        return descriptor