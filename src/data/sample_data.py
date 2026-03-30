"""
Generate synthetic but realistic options data for testing.

Creates underlying price data (geometric Brownian motion)
and a corresponding options chain with realistic Greeks.
Uses TIMESTAMP instead of DATE for resolution-agnostic support.

Usage:
    python -m src.data.sample_data --ticker AAPL --start 2023-01-01 --end 2024-01-01
"""

import math
import random
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def _round2(val: float) -> Decimal:
    return Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _round4(val: float) -> Decimal:
    return Decimal(str(val)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return (1.0 + erf(x / sqrt(2.0))) / 2.0


def _bs_delta(S, K, T, r, sigma, opt_type):
    if T <= 0:
        if opt_type == "C":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    from math import log, sqrt
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return _norm_cdf(d1) if opt_type == "C" else _norm_cdf(d1) - 1.0


def _bs_price(S, K, T, r, sigma, opt_type):
    if T <= 0:
        return max(S - K, 0) if opt_type == "C" else max(K - S, 0)
    from math import log, sqrt, exp
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if opt_type == "C":
        return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    else:
        return K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _bs_gamma(S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    from math import log, sqrt, exp, pi
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return exp(-d1**2 / 2) / (S * sigma * sqrt(2 * pi * T))


def _bs_theta(S, K, T, r, sigma, opt_type):
    if T <= 0:
        return 0.0
    from math import log, sqrt, exp, pi
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    common = -(S * sigma * exp(-d1**2 / 2)) / (2 * sqrt(2 * pi * T))
    if opt_type == "C":
        return (common - r * K * exp(-r * T) * _norm_cdf(d2)) / 365
    else:
        return (common + r * K * exp(-r * T) * _norm_cdf(-d2)) / 365


def _bs_vega(S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    from math import log, sqrt, exp, pi
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return S * sqrt(T) * exp(-d1**2 / 2) / sqrt(2 * pi) / 100


def generate_underlying(
    ticker: str,
    start: date,
    end: date,
    initial_price: float = 180.0,
    annual_return: float = 0.10,
    annual_vol: float = 0.25,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic daily OHLCV data using geometric Brownian motion."""
    random.seed(seed)
    rows = []
    price = initial_price
    dt = 1 / 252
    current = start

    while current <= end:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        drift = (annual_return - 0.5 * annual_vol**2) * dt
        shock = annual_vol * math.sqrt(dt) * random.gauss(0, 1)
        price *= math.exp(drift + shock)

        intraday_vol = price * annual_vol / math.sqrt(252) * 0.6
        high = price + abs(random.gauss(0, intraday_vol))
        low = price - abs(random.gauss(0, intraday_vol))
        open_price = low + random.random() * (high - low)

        volume = max(int(random.gauss(50_000_000, 15_000_000)), 1_000_000)

        # Use TIMESTAMP (midnight) for daily bars
        ts = datetime(current.year, current.month, current.day)

        rows.append({
            "ticker": ticker,
            "timestamp": ts,
            "open": _round2(open_price),
            "high": _round2(high),
            "low": _round2(low),
            "close": _round2(price),
            "volume": volume,
            "dividend": Decimal("0"),
        })

        current += timedelta(days=1)

    return rows


def generate_options_chain(
    underlying_rows: list[dict],
    risk_free_rate: float = 0.05,
    base_iv: float = 0.25,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic options chain for each trading day."""
    random.seed(seed)
    rows = []

    all_dates = [r["timestamp"] for r in underlying_rows]
    start_year = all_dates[0].year
    end_year = all_dates[-1].year + 2

    # Third Friday expirations
    expirations = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            first_day = date(year, month, 1)
            weekday = first_day.weekday()
            first_friday = first_day + timedelta(days=(4 - weekday) % 7)
            third_friday = first_friday + timedelta(days=14)
            expirations.append(third_friday)

    for urow in underlying_rows:
        S = float(urow["close"])
        current_ts = urow["timestamp"]
        current_date = current_ts.date() if isinstance(current_ts, datetime) else current_ts
        ticker = urow["ticker"]

        valid_exps = [
            exp for exp in expirations
            if 7 <= (exp - current_date).days <= 730
        ]

        step = 2.5 if S < 200 else 5.0
        min_strike = math.floor(S * 0.7 / step) * step
        max_strike = math.ceil(S * 1.3 / step) * step
        strikes = []
        k = min_strike
        while k <= max_strike:
            strikes.append(k)
            k += step

        for exp in valid_exps:
            T = (exp - current_date).days / 365.0

            for K in strikes:
                moneyness = math.log(K / S)
                iv = base_iv + 0.1 * moneyness**2 + random.gauss(0, 0.01)
                iv = max(iv, 0.05)

                for opt_type in ("C", "P"):
                    theo_price = _bs_price(S, K, T, risk_free_rate, iv, opt_type)
                    if theo_price < 0.01:
                        continue

                    delta = _bs_delta(S, K, T, risk_free_rate, iv, opt_type)
                    gamma = _bs_gamma(S, K, T, risk_free_rate, iv)
                    theta = _bs_theta(S, K, T, risk_free_rate, iv, opt_type)
                    vega = _bs_vega(S, K, T, risk_free_rate, iv)

                    spread_pct = 0.03 + 0.02 * (1 - abs(delta))
                    half_spread = theo_price * spread_pct
                    bid = max(theo_price - half_spread, 0.01)
                    ask = theo_price + half_spread

                    base_vol = int(5000 * math.exp(-2 * moneyness**2))
                    oi = int(base_vol * random.uniform(5, 20))

                    rows.append({
                        "ticker": ticker,
                        "timestamp": current_ts,  # TIMESTAMP, not date
                        "expiration": exp,
                        "strike": _round2(K),
                        "option_type": opt_type,
                        "bid": _round2(bid),
                        "ask": _round2(ask),
                        "last": _round2(theo_price + random.gauss(0, half_spread * 0.3)),
                        "volume": max(int(random.gauss(base_vol, base_vol * 0.5)), 0),
                        "open_interest": max(oi, 0),
                        "implied_vol": _round4(iv),
                        "delta": _round4(delta),
                        "gamma": _round4(gamma),
                        "theta": _round4(theta),
                        "vega": _round4(vega),
                    })

    return rows


def generate_and_save(
    ticker: str = "AAPL",
    start: str = "2023-01-01",
    end: str = "2024-01-01",
    output_dir: str = "data/processed",
    initial_price: float = 180.0,
):
    """Generate sample data and save as Parquet files."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    print(f"Generating underlying data for {ticker}...")
    underlying = generate_underlying(ticker, start_date, end_date, initial_price=initial_price)

    print(f"Generating options chain ({len(underlying)} trading days)...")
    chain = generate_options_chain(underlying)
    print(f"  Generated {len(chain):,} option records")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save underlying
    df_underlying = pd.DataFrame(underlying)
    for col in ["open", "high", "low", "close", "dividend"]:
        df_underlying[col] = df_underlying[col].astype(float)
    pq.write_table(
        pa.Table.from_pandas(df_underlying),
        output_path / f"underlying_{ticker.lower()}.parquet",
    )
    print(f"  Saved underlying_{ticker.lower()}.parquet")

    # Save options chain
    df_chain = pd.DataFrame(chain)
    decimal_cols = ["strike", "bid", "ask", "last", "implied_vol", "delta", "gamma", "theta", "vega"]
    for col in decimal_cols:
        df_chain[col] = df_chain[col].astype(float)
    pq.write_table(
        pa.Table.from_pandas(df_chain),
        output_path / f"options_{ticker.lower()}.parquet",
    )
    print(f"  Saved options_{ticker.lower()}.parquet")
    print("Done!")


if __name__ == "__main__":
    import typer
    typer.run(generate_and_save)
