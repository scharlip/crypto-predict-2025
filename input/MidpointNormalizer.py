from typing import List

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class MidpointNormalizer:

    def __init__(self, scaler: MinMaxScaler):
        self.scaler = scaler

    def normalize(self, vals: List[float]):
        log10 = np.log10(vals)
        scaled = self.scaler.fit_transform(log10)
        return scaled

    def denormalize(self, vals: List[float]):
        unscaled = self.scaler.inverse_transform([vals])
        exp = [(10.0 ** v) for v in unscaled[0].tolist()]
        return exp
