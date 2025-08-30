from enum import StrEnum

class CoinType(StrEnum):
    BTC = "BTC"
    ETH = "ETH"
    ADA = "ADA"

class Exchange(StrEnum):
    Binance = "Binance"
    Bitfinex = "Bitfinex"
    BitMEX = "BitMEX"
    Bitstamp = "Bitstamp"
    Coinbase = "Coinbase"
    KuCoin = "KuCoin"