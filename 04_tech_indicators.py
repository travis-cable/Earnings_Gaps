# Originally written by Jackson Walker (https://github.com/jacksonrgwalker), modified by Travis Cable

import pandas as pd
from pathlib import Path
from stockstats import wrap


def calculate_technical_indicators(ohlc: pd.DataFrame) -> pd.DataFrame:

    wrap_df = wrap(ohlc)

    tech_indicator_cols = [
        "adx",
        "trix",
        "adxr",
        "cci",
        "macd",
        "macdh",
        "rsi_14",
        "kdjk",
        "wr_14",
        "atr",
    ]

    wrap_df[tech_indicator_cols]

    wrap_df["clv"] = ((wrap_df["close"] - wrap_df["low"]) - (wrap_df["high"] - wrap_df["close"])) / (
            wrap_df["high"] - wrap_df["low"]
    )
    wrap_df["cmf"] = wrap_df["clv"] * wrap_df["volume"]
    wrap_df["cmf"] = (
            wrap_df["cmf"].rolling(window=20).sum() / wrap_df["volume"].rolling(window=20).sum()
    )
    wrap_df["atr_percent"] = wrap_df["atr"] / (wrap_df["high"] - wrap_df["low"])

    return wrap_df


ohlc_save_path = Path("data/ohlc.parquet")
ohlc = pd.read_parquet(ohlc_save_path)

technical = ohlc.groupby("ticker").apply(calculate_technical_indicators)
technical = (technical.reset_index(level=0, drop=True)).reset_index(level=0, drop=True)
technical.sort_index(inplace=True)

technical_save_path = Path("data/technical_indicators.parquet")
technical.to_parquet(technical_save_path)
