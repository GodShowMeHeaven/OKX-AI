import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from trading_bot import OKXFuturesBot


def _generate_candles(count: int = 60):
    candles = []
    base_price = 100
    for i in range(count):
        price = base_price + i
        candles.append([
            str(i),            # ts
            str(price),        # o
            str(price + 1),    # h
            str(price - 1),    # l
            str(price),        # c
            "100",             # vol
            "0",               # volCcy
            "0",               # volCcyQuote
            "1",               # confirm
        ])
    return candles


def test_calculate_indicators_returns_numeric_values():
    bot = OKXFuturesBot("api", "secret", "pass", "openai")
    bot.candles_1m = _generate_candles()

    indicators = bot.calculate_indicators()

    for key in ("rsi", "ema_20", "macd_histogram"):
        assert key in indicators
        value = indicators[key]
        assert isinstance(value, (int, float, np.number))
        assert not np.isnan(value)
