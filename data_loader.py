"""
data_loader.py - Historical OHLCV ingestion, validation and persistence.

Pipeline
--------
    CCXT REST (paginated, rate-limited, retried with exponential back-off)
        -> validation & cleaning (duplicates, impossible candles, gap report)
        -> SQLite cache (idempotent UPSERT, incremental refresh)
        -> pandas DataFrame with a UTC DatetimeIndex and float64 OHLCV columns

Also provided: CSV import/export, optional Parquet export, and a regime-switching
GBM generator that produces realistic synthetic crypto candles for offline research
and unit tests (no network or exchange account required).

Conventions
-----------
* Each row is one *closed* candle; the index is the candle's **open** time in UTC.
* Missing candles are reported but never fabricated. Forward-filling a gap would
  create zero-volatility bars that flatter risk metrics.
"""

from __future__ import annotations

import logging
import random
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import MINUTES_PER_YEAR, AppConfig, DataSource, ExchangeConfig, timeframe_to_minutes

logger = logging.getLogger(__name__)

OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


class DataLoaderError(RuntimeError):
    """Raised when market data cannot be fetched, parsed or validated."""


# --------------------------------------------------------------------------------------
# Timestamp helpers (resolution-agnostic: work for ns/us/ms-based DatetimeIndex)
# --------------------------------------------------------------------------------------
def to_epoch_ms(index: pd.DatetimeIndex) -> np.ndarray:
    """UTC DatetimeIndex -> int64 milliseconds since the Unix epoch."""
    return np.asarray((index - _EPOCH) // pd.Timedelta(milliseconds=1), dtype="int64")


def from_epoch_ms(values: Sequence[int] | np.ndarray) -> pd.DatetimeIndex:
    """int milliseconds since the epoch -> UTC DatetimeIndex named ``timestamp``."""
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(values, dtype="int64"), unit="ms", utc=True), name="timestamp")


def ohlcv_frame(rows: Sequence[Sequence[float]]) -> pd.DataFrame:
    """Convert CCXT's ``[[ts_ms, open, high, low, close, volume], ...]`` into a DataFrame."""
    if not rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"), dtype=float)
    arr = np.asarray([r[:6] for r in rows], dtype="float64")
    frame = pd.DataFrame(arr[:, 1:6], columns=OHLCV_COLUMNS, index=from_epoch_ms(arr[:, 0].astype("int64")))
    return frame


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    duplicates_removed: int
    invalid_rows_removed: int
    missing_bars: int
    largest_gap: pd.Timedelta
    zero_volume_bars: int

    def summary(self) -> str:
        if not self.rows:
            return "Data quality: empty dataset"
        return (
            f"Data quality: {self.rows:,} bars {self.start:%Y-%m-%d %H:%M} -> {self.end:%Y-%m-%d %H:%M} UTC | "
            f"duplicates removed={self.duplicates_removed} invalid removed={self.invalid_rows_removed} "
            f"missing bars={self.missing_bars} (largest gap {self.largest_gap}) zero-volume={self.zero_volume_bars}"
        )


def validate_and_clean(df: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, DataQualityReport]:
    """Sort, de-duplicate and sanity-check OHLCV candles.

    Rows are dropped when prices are non-positive/NaN, volume is negative, or the
    candle is geometrically impossible (high below the body or low above it).
    """
    missing_cols = set(OHLCV_COLUMNS) - set(df.columns)
    if missing_cols:
        raise DataLoaderError(f"OHLCV data is missing columns: {sorted(missing_cols)}")

    out = df[OHLCV_COLUMNS].astype("float64").sort_index()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    out.index.name = "timestamp"

    dup_mask = out.index.duplicated(keep="last")  # the latest update of a candle wins
    duplicates = int(dup_mask.sum())
    out = out[~dup_mask]

    body_hi = out[["open", "close"]].max(axis=1)
    body_lo = out[["open", "close"]].min(axis=1)
    valid = (
        out[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & out["volume"].ge(0)
        & out["high"].ge(body_hi)
        & out["low"].le(body_lo)
        & out.notna().all(axis=1)
    )
    invalid = int((~valid).sum())
    out = out[valid]

    bar = pd.Timedelta(minutes=timeframe_to_minutes(timeframe))
    missing, largest_gap = 0, pd.Timedelta(0)
    if len(out) > 1:
        deltas = out.index.to_series().diff().dropna()
        gaps = deltas[deltas > bar]
        missing = int(((gaps / bar).round() - 1).sum())
        largest_gap = gaps.max() if not gaps.empty else pd.Timedelta(0)
    if missing:
        logger.warning("Detected %d missing bars (largest gap %s); gaps are left unfilled.", missing, largest_gap)

    report = DataQualityReport(
        rows=len(out),
        start=out.index[0] if len(out) else None,
        end=out.index[-1] if len(out) else None,
        duplicates_removed=duplicates,
        invalid_rows_removed=invalid,
        missing_bars=missing,
        largest_gap=largest_gap,
        zero_volume_bars=int(out["volume"].eq(0).sum()),
    )
    return out, report


# --------------------------------------------------------------------------------------
# SQLite persistence
# --------------------------------------------------------------------------------------
class OHLCVStore:
    """SQLite-backed candle cache keyed by (exchange, symbol, timeframe, ts)."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS ohlcv (
            exchange  TEXT    NOT NULL,
            symbol    TEXT    NOT NULL,
            timeframe TEXT    NOT NULL,
            ts        INTEGER NOT NULL,          -- candle open time, ms since epoch (UTC)
            open      REAL    NOT NULL,
            high      REAL    NOT NULL,
            low       REAL    NOT NULL,
            close     REAL    NOT NULL,
            volume    REAL    NOT NULL,
            PRIMARY KEY (exchange, symbol, timeframe, ts)
        ) WITHOUT ROWID;
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(self._SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")    # concurrent readers while writing
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def upsert(self, df: pd.DataFrame, exchange: str, symbol: str, timeframe: str) -> int:
        """Insert or update candles. Idempotent, so re-downloading overlaps is harmless."""
        if df.empty:
            return 0
        ts = to_epoch_ms(pd.DatetimeIndex(df.index)).tolist()
        cols = [df[c].astype(float).tolist() for c in OHLCV_COLUMNS]
        rows = [(exchange, symbol, timeframe, t, *vals) for t, *vals in zip(ts, *cols)]
        sql = (
            "INSERT INTO ohlcv (exchange, symbol, timeframe, ts, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(exchange, symbol, timeframe, ts) DO UPDATE SET "
            "open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume"
        )
        with closing(self._connect()) as conn:
            with conn:  # transaction: commit on success, rollback on error
                conn.executemany(sql, rows)
        logger.debug("Upserted %d candles for %s %s %s", len(rows), exchange, symbol, timeframe)
        return len(rows)

    def load(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        sql = "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE exchange=? AND symbol=? AND timeframe=?"
        params: list[Any] = [exchange, symbol, timeframe]
        if start is not None:
            sql += " AND ts >= ?"
            params.append(int(to_epoch_ms(pd.DatetimeIndex([start]))[0]))
        if end is not None:
            sql += " AND ts <= ?"
            params.append(int(to_epoch_ms(pd.DatetimeIndex([end]))[0]))
        sql += " ORDER BY ts"
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return ohlcv_frame(rows)

    def bounds(self, exchange: str, symbol: str, timeframe: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """(first, last) stored candle timestamps, or (None, None) when nothing is cached."""
        with closing(self._connect()) as conn:
            lo, hi = conn.execute(
                "SELECT MIN(ts), MAX(ts) FROM ohlcv WHERE exchange=? AND symbol=? AND timeframe=?",
                (exchange, symbol, timeframe),
            ).fetchone()
        if lo is None:
            return None, None
        idx = from_epoch_ms([lo, hi])
        return idx[0], idx[1]

    def catalogue(self) -> pd.DataFrame:
        """Summary of every cached series."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT exchange, symbol, timeframe, COUNT(*), MIN(ts), MAX(ts) FROM ohlcv "
                "GROUP BY exchange, symbol, timeframe ORDER BY exchange, symbol, timeframe"
            ).fetchall()
        cat = pd.DataFrame(rows, columns=["exchange", "symbol", "timeframe", "bars", "first", "last"])
        if not cat.empty:
            cat["first"] = from_epoch_ms(cat["first"].to_numpy())
            cat["last"] = from_epoch_ms(cat["last"].to_numpy())
        return cat


# --------------------------------------------------------------------------------------
# File formats
# --------------------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="timestamp", date_format="%Y-%m-%dT%H:%M:%SZ")
    return path


def load_csv(path: Path) -> pd.DataFrame:
    """Load OHLCV from CSV. Accepts ISO dates or epoch-millisecond timestamps."""
    path = Path(path)
    if not path.is_file():
        raise DataLoaderError(f"CSV file not found: {path}")
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    time_col = next((c for c in ("timestamp", "datetime", "date", "time", "open_time") if c in raw.columns), None)
    if time_col is None:
        raise DataLoaderError(f"No timestamp column found in {path.name} (expected 'timestamp' or 'date')")
    col = raw[time_col]
    if pd.api.types.is_numeric_dtype(col):
        index = from_epoch_ms(col.astype("int64").to_numpy())
    else:
        index = pd.DatetimeIndex(pd.to_datetime(col, utc=True), name="timestamp")
    missing_cols = set(OHLCV_COLUMNS) - set(raw.columns)
    if missing_cols:
        raise DataLoaderError(f"CSV {path.name} is missing columns: {sorted(missing_cols)}")
    return pd.DataFrame(raw[OHLCV_COLUMNS].to_numpy(dtype="float64"), index=index, columns=OHLCV_COLUMNS)


def save_parquet(df: pd.DataFrame, path: Path) -> Path | None:
    """Columnar export for large datasets. Optional: needs a working pyarrow/fastparquet."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return path
    except Exception as exc:  # noqa: BLE001 - engine missing or its native library blocked
        logger.warning("Parquet export skipped (%s). SQLite/CSV storage is unaffected.", exc)
        return None


# --------------------------------------------------------------------------------------
# Exchange download (CCXT REST)
# --------------------------------------------------------------------------------------
class ExchangeDataFetcher:
    """Paginated OHLCV download through CCXT's unified REST API.

    Public market data needs no API key, so the fetcher always uses mainnet endpoints
    (testnets have thin, unrealistic histories).
    """

    def __init__(self, cfg: ExchangeConfig) -> None:
        self.cfg = cfg
        self._exchange: Any = None
        self._markets_loaded = False

    @property
    def exchange(self) -> Any:
        if self._exchange is None:
            try:
                import ccxt  # imported lazily so offline modes work without it
            except ImportError as exc:
                raise DataLoaderError(
                    "The 'ccxt' package is required for exchange data. Install it with `pip install ccxt`, "
                    "or run with --source synthetic / --source csv."
                ) from exc
            if not hasattr(ccxt, self.cfg.exchange_id):
                raise DataLoaderError(f"Unknown CCXT exchange id '{self.cfg.exchange_id}'")
            self._exchange = getattr(ccxt, self.cfg.exchange_id)(self.cfg.ccxt_options(authenticated=False))
        return self._exchange

    def _call_with_retry(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Retry transient network failures with exponential back-off and jitter.

        delay_k = base * 2^(k-1) * U(0.5, 1.0). The jitter de-synchronises clients
        that failed at the same moment (the "thundering herd" problem).
        """
        import ccxt

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except ccxt.NetworkError as exc:  # timeouts, 5xx, DDoS protection, rate limits
                if attempt == self.cfg.max_retries:
                    raise DataLoaderError(
                        f"{self.cfg.exchange_id} unreachable after {attempt} attempts: {exc}. "
                        "If the exchange is geo-restricted where you are, set EXCHANGE_ID to another venue "
                        "(e.g. binanceus, kraken, okx, bybit)."
                    ) from exc
                delay = min(30.0, 1.0 * 2 ** (attempt - 1)) * random.uniform(0.5, 1.0)
                logger.warning("Network error (%s). Retry %d/%d in %.1fs", type(exc).__name__, attempt,
                               self.cfg.max_retries, delay)
                time.sleep(delay)
            except ccxt.ExchangeError as exc:  # bad symbol, permissions: retrying will not help
                raise DataLoaderError(f"Exchange rejected the request: {exc}") from exc
        raise DataLoaderError("unreachable")  # pragma: no cover

    def _ensure_market(self, symbol: str) -> None:
        if not self._markets_loaded:
            self._call_with_retry(self.exchange.load_markets)
            self._markets_loaded = True
        if symbol not in self.exchange.markets:
            raise DataLoaderError(f"Symbol '{symbol}' is not listed on {self.cfg.exchange_id}")
        if not self.exchange.has.get("fetchOHLCV"):
            raise DataLoaderError(f"{self.cfg.exchange_id} does not support OHLCV downloads")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: pd.Timestamp,
        until: pd.Timestamp | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Download closed candles in [since, until) by walking forward page by page."""
        self._ensure_market(symbol)
        tf_ms = timeframe_to_minutes(timeframe) * 60_000
        now_ms = int(self.exchange.milliseconds())
        cursor = int(since.timestamp() * 1000)
        until_ms = min(int(until.timestamp() * 1000), now_ms) if until is not None else now_ms

        rows: list[list[float]] = []
        pages = 0
        while cursor < until_ms:
            batch = self._call_with_retry(self.exchange.fetch_ohlcv, symbol, timeframe, since=cursor, limit=limit)
            if not batch:
                break
            rows.extend(batch)
            pages += 1
            last_ts = int(batch[-1][0])
            if last_ts < cursor:  # defensive: the exchange ignored `since`
                break
            cursor = last_ts + tf_ms
            if pages % 10 == 0:
                logger.info("  ... %d candles downloaded (up to %s)", len(rows), from_epoch_ms([last_ts])[0])

        df = ohlcv_frame(rows)
        if df.empty:
            return df
        df = df[~df.index.duplicated(keep="last")]
        ts = to_epoch_ms(pd.DatetimeIndex(df.index))
        # Keep closed candles only: the newest candle is still forming until open + tf.
        closed = (ts + tf_ms <= now_ms) & (ts < until_ms)
        return df[closed]


# --------------------------------------------------------------------------------------
# Synthetic market generator
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketRegime:
    name: str
    drift: float       # annualised mu
    volatility: float  # annualised sigma


class SyntheticDataGenerator:
    """Regime-switching Geometric Brownian Motion with Merton jumps.

    Log-return of bar t in hidden regime k:
        r_t = (mu_k - sigma_k^2 / 2) * dt + sigma_k * sqrt(dt) * Z_t + J_t,   Z_t ~ N(0, 1)
        J_t = sum of N_t jumps, N_t ~ Poisson(lambda * dt), each jump ~ N(mu_J, sigma_J^2)
    Regime k follows a Markov chain with persistence p_stay (mean regime length
    1 / (1 - p_stay) bars), which produces the alternating trends and ranges that
    trend-following systems such as Ichimoku are designed for.

    High/low: wicks extend beyond the candle body by |N(0,1)| * sigma_bar * 0.6, a
    cheap stand-in for the Brownian-bridge range. Volume is lognormal and rises with
    the absolute return (the volume-volatility correlation seen in real markets).
    """

    REGIMES: tuple[MarketRegime, ...] = (
        MarketRegime("bull", drift=0.90, volatility=0.55),
        MarketRegime("bear", drift=-0.70, volatility=0.75),
        MarketRegime("range", drift=0.0, volatility=0.35),
    )

    def __init__(
        self,
        seed: int = 42,
        p_stay: float = 0.998,
        jump_intensity: float = 12.0,
        jump_mean: float = -0.005,
        jump_std: float = 0.03,
        base_volume: float = 800.0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.p_stay = p_stay
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.base_volume = base_volume

    def generate(
        self,
        n_bars: int,
        timeframe: str = "1h",
        start: pd.Timestamp | None = None,
        s0: float = 30_000.0,
    ) -> pd.DataFrame:
        if n_bars < 2:
            raise DataLoaderError("n_bars must be >= 2")
        minutes = timeframe_to_minutes(timeframe)
        dt = minutes / MINUTES_PER_YEAR
        rng = self.rng
        k = len(self.REGIMES)

        # Markov regime chain (sequential by nature; O(n) and fast enough for 10^6 bars).
        switch = rng.random(n_bars) > self.p_stay
        hop = rng.integers(1, k, n_bars)
        regimes = np.empty(n_bars, dtype=np.int64)
        state = 0
        for i in range(n_bars):
            if switch[i]:
                state = (state + hop[i]) % k
            regimes[i] = state

        mu = np.array([r.drift for r in self.REGIMES])[regimes]
        sigma = np.array([r.volatility for r in self.REGIMES])[regimes]
        n_jumps = rng.poisson(self.jump_intensity * dt, n_bars)
        jumps = n_jumps * self.jump_mean + np.sqrt(n_jumps) * self.jump_std * rng.standard_normal(n_bars)
        log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_bars) + jumps

        close = s0 * np.exp(np.cumsum(log_ret))
        open_ = np.concatenate(([s0], close[:-1]))
        bar_sigma = sigma * np.sqrt(dt)
        high = np.maximum(open_, close) * np.exp(np.abs(rng.standard_normal(n_bars)) * bar_sigma * 0.6)
        low = np.minimum(open_, close) * np.exp(-np.abs(rng.standard_normal(n_bars)) * bar_sigma * 0.6)
        volume = (
            self.base_volume * (minutes / 60)
            * np.exp(0.35 * rng.standard_normal(n_bars))
            * (1.0 + 1.5 * np.abs(log_ret) / bar_sigma)
        )

        step = pd.Timedelta(minutes=minutes)
        if start is None:
            last_open = pd.Timestamp.now(tz="UTC").floor(f"{minutes}min") - step
            start = last_open - step * (n_bars - 1)
        index = pd.date_range(start=start, periods=n_bars, freq=step, name="timestamp")
        if index.tz is None:
            index = index.tz_localize("UTC")
        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


# --------------------------------------------------------------------------------------
# Facade
# --------------------------------------------------------------------------------------
class DataLoader:
    """Single entry point used by the CLI, backtester and live engine."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.store = OHLCVStore(cfg.data.db_path)
        self._fetcher: ExchangeDataFetcher | None = None
        self.last_report: DataQualityReport | None = None

    @property
    def fetcher(self) -> ExchangeDataFetcher:
        if self._fetcher is None:
            self._fetcher = ExchangeDataFetcher(self.cfg.exchange)
        return self._fetcher

    def load(self, source: DataSource | None = None, refresh: bool = True) -> pd.DataFrame:
        """Return validated OHLCV for the configured symbol/timeframe/history window."""
        d = self.cfg.data
        source = DataSource(source or d.source)
        if source is DataSource.EXCHANGE:
            raw = self._load_exchange(refresh)
        elif source is DataSource.SYNTHETIC:
            n_bars = int(d.history_days * 1440 / timeframe_to_minutes(d.timeframe))
            logger.info("Generating %d synthetic %s bars (seed=%d). Results are NOT real market data.",
                        n_bars, d.timeframe, d.synthetic_seed)
            raw = SyntheticDataGenerator(seed=d.synthetic_seed).generate(n_bars, d.timeframe)
        elif source is DataSource.CSV:
            if d.csv_path is None:
                raise DataLoaderError("CSV source selected but no csv_path configured (use --csv-path)")
            raw = load_csv(d.csv_path)
        else:  # pragma: no cover - enum exhausted
            raise DataLoaderError(f"Unsupported data source {source}")

        df, report = validate_and_clean(raw, d.timeframe)
        self.last_report = report
        logger.info(report.summary())
        if len(df) < d.min_bars:
            raise DataLoaderError(f"Only {len(df)} valid bars available; at least {d.min_bars} are required")
        return df

    def _load_exchange(self, refresh: bool) -> pd.DataFrame:
        d, ex_id = self.cfg.data, self.cfg.exchange.exchange_id
        bar = pd.Timedelta(minutes=timeframe_to_minutes(d.timeframe))
        start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=d.history_days)
        first, last = self.store.bounds(ex_id, d.symbol, d.timeframe)

        if refresh:
            # Incremental refresh: only download what the cache does not hold yet.
            since = start if first is None or first > start + bar else last + bar
            logger.info("Fetching %s %s from %s since %s", d.symbol, d.timeframe, ex_id, f"{since:%Y-%m-%d %H:%M}")
            try:
                fresh = self.fetcher.fetch_ohlcv(d.symbol, d.timeframe, since=since)
                if not fresh.empty:
                    fresh, _ = validate_and_clean(fresh, d.timeframe)
                stored = self.store.upsert(fresh, ex_id, d.symbol, d.timeframe)
                logger.info("Stored %d new/updated candles in %s", stored, self.store.db_path.name)
            except DataLoaderError as exc:
                if first is None:
                    raise
                logger.warning("Download failed (%s); using cached data %s -> %s", exc, first, last)

        df = self.store.load(ex_id, d.symbol, d.timeframe, start=start)
        if df.empty:
            raise DataLoaderError(f"No cached data for {d.symbol} {d.timeframe} on {ex_id}")
        return df

    def export(self, df: pd.DataFrame, directory: Path | None = None) -> dict[str, Path]:
        """Write the dataset to CSV (and Parquet when available)."""
        directory = Path(directory or self.cfg.data.db_path.parent)
        stem = f"{self.cfg.data.symbol.replace('/', '')}_{self.cfg.data.timeframe}"
        paths = {"csv": save_csv(df, directory / f"{stem}.csv")}
        parquet = save_parquet(df, directory / f"{stem}.parquet")
        if parquet is not None:
            paths["parquet"] = parquet
        return paths
