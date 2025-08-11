import torch
from torch import optim, nn

from common.common import CoinType, Exchange
from input.input import MidpointCoinDataset
from models.models import SimpleLSTMModel
from train.train import train_loop

ds = MidpointCoinDataset(coin_type=CoinType.ETH, exchange=Exchange.Coinbase)
device = torch.device("cpu")
model = SimpleLSTMModel(input_size=ds.lookback_window_size, device=device)

train_loop(
    ds=ds,
    model=model,
    device=device,
    optimizer=optim.Adam(model.parameters()),
    loss_fn = nn.MSELoss()
)