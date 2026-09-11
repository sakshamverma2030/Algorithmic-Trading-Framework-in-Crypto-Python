"""
config.py - Global configuration for the Ichimoku Cloud crypto trading system.

Every tunable parameter lives here as an immutable (frozen) dataclass, so a single
``AppConfig`` object fully describes an experiment. That gives us two properties a
production trading system needs:

* Reproducibility - the config is serialised next to every backtest report.
* Research/production parity - the backtester and the live engine read the *same*
  objects, so the strategy that was tested is the strategy that trades.

Secrets (exchange API keys) are never hard-coded. They are read from environment
variables, or from a local ``.env`` file (git-ignored) that is parsed at import time.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
LOG_DIR: Path = BASE_DIR / "logs"
REPORT_DIR: Path = BASE_DIR / "reports"
MARKET_DB_PATH: Path = DATA_DIR / "market_data.sqlite"
JOURNAL_DB_PATH: Path = DATA_DIR / "trade_journal.sqlite"


class ConfigError(ValueError):
    """Raised when a configuration value is invalid or inconsistent."""


# --------------------------------------------------------------------------------------
# Environment helpers
# --------------------------------------------------------------------------------------
def _load_dotenv(path: Path = BASE_DIR / ".env") -> None:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into ``os.environ``.

    A deliberately tiny parser (no third-party dependency). Variables that already
    exist in the process environment take precedence over the file.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


_load_dotenv()


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------
class TradingMode(str, Enum):
    """Execution mode of the real-time engine."""

    PAPER_TRADING = "PAPER_TRADING"  # simulated fills against the live order book
    LIVE_TRADING = "LIVE_TRADING"    # real orders through the exchange API (keys required)


class SizingMethod(str, Enum):
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY = "kelly"


class StopMethod(str, Enum):
    ATR = "atr"      # stop = entry -/+ k * ATR
    CLOUD = "cloud"  # stop = opposite Kumo boundary -/+ buffer * ATR


class TrailingMethod(str, Enum):
    NONE = "none"
    ATR = "atr"      # Chandelier exit: extreme price since entry -/+ m * ATR
    KIJUN = "kijun"  # classic Ichimoku trailing stop on the Kijun-sen


class DataSource(str, Enum):
    EXCHANGE = "exchange"    # CCXT REST download, cached in SQLite
    SYNTHETIC = "synthetic"  # regime-switching GBM (offline research / tests)
    CSV = "csv"              # user-supplied OHLCV file


class FeedType(str, Enum):
    CCXT_PRO = "ccxtpro"  # WebSocket streams (ccxt.pro, bundled with ccxt >= 4)
    REST = "rest"         # REST polling fallback
    REPLAY = "replay"     # historical bar replay through the live engine (offline demo)


# --------------------------------------------------------------------------------------
# Timeframes & annualisation
# --------------------------------------------------------------------------------------
TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720, "1d": 1440,
}

# Crypto trades 24 hours a day, 365 days a year - no weekends, no holidays. The
# annualisation factor therefore uses the full calendar, unlike equities (252 days).
MINUTES_PER_YEAR: int = 365 * 24 * 60


def timeframe_to_minutes(timeframe: str) -> int:
    """Return the bar length in minutes for a CCXT timeframe string."""
    try:
        return TIMEFRAME_MINUTES[timeframe]
    except KeyError as exc:
        raise ConfigError(
            f"Unsupported timeframe '{timeframe}'. Choose one of: {', '.join(TIMEFRAME_MINUTES)}"
        ) from exc


def periods_per_year(timeframe: str) -> float:
    """Annualisation factor N = 525,600 / bar_minutes (1h -> 8,760 ; 15m -> 35,040)."""
    return MINUTES_PER_YEAR / timeframe_to_minutes(timeframe)


# --------------------------------------------------------------------------------------
# Strategy parameters
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class IchimokuParams:
    """Look-back periods of the Ichimoku Kinko Hyo system.

    Goichi Hosoda's original 9/26/52 settings were calibrated to the six-day Japanese
    trading week of the 1930s: 9 = 1.5 weeks, 26 = one month, 52 = two months.
    """

    tenkan: int = 9         # Conversion line window
    kijun: int = 26         # Base line window
    senkou_b: int = 52      # Leading span B window
    displacement: int = 26  # forward shift of the Kumo / backward shift of the Chikou

    def __post_init__(self) -> None:
        if not 0 < self.tenkan < self.kijun < self.senkou_b:
            raise ConfigError(
                f"Ichimoku periods must satisfy 0 < tenkan < kijun < senkou_b (got {self.label})"
            )
        if self.displacement <= 0:
            raise ConfigError("Ichimoku displacement must be positive")

    @property
    def label(self) -> str:
        return f"{self.tenkan}/{self.kijun}/{self.senkou_b}/{self.displacement}"

    @property
    def warmup(self) -> int:
        """Bars required before the displaced Senkou Span B has its first value."""
        return self.senkou_b + self.displacement


ICHIMOKU_PRESETS: dict[str, IchimokuParams] = {
    # Hosoda's original settings (six-day trading week).
    "standard": IchimokuParams(9, 26, 52, 26),
    # 24/7 calendar adjustment: 1.5 weeks = 10 days, 1 month = 30, 2 months = 60.
    "crypto": IchimokuParams(10, 30, 60, 30),
    # Doubled periods, popular on noisy intraday crypto charts (fewer whipsaws).
    "crypto_slow": IchimokuParams(20, 60, 120, 30),
}


@dataclass(frozen=True)
class StrategyConfig:
    """Signal-generation parameters for the Ichimoku strategy."""

    params: IchimokuParams = field(default_factory=lambda: ICHIMOKU_PRESETS["standard"])
    # Dynamic mode: switch between presets by volatility regime (see indicators.AdaptiveIchimoku).
    dynamic: bool = False
    regime_presets: tuple[str, str, str] = ("standard", "crypto", "crypto_slow")  # low / normal / high vol
    regime_lookback: int = 500          # bars in the rolling volatility percentile window
    regime_low_quantile: float = 0.33
    regime_high_quantile: float = 0.67
    atr_period: int = 14
    allow_short: bool = False           # spot markets are long-only; enable for perps/margin
    cross_lookback: int = 1             # bars a TK cross stays "fresh" (1 = cross on the signal bar)
    require_chikou: bool = True         # Chikou confirmation: Close_t vs Close_{t-displacement}
    require_kumo_twist: bool = False    # optional filter: future (leading) cloud has the trade's colour

    def __post_init__(self) -> None:
        unknown = [p for p in self.regime_presets if p not in ICHIMOKU_PRESETS]
        if unknown:
            raise ConfigError(f"Unknown regime preset(s): {unknown}")
        if not 0 < self.regime_low_quantile < self.regime_high_quantile < 1:
            raise ConfigError("Regime quantiles must satisfy 0 < low < high < 1")
        if self.cross_lookback < 1 or self.atr_period < 2 or self.regime_lookback < 20:
            raise ConfigError("cross_lookback >= 1, atr_period >= 2 and regime_lookback >= 20 required")

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> StrategyConfig:
        if name == "dynamic":
            return cls(dynamic=True, **overrides)
        if name not in ICHIMOKU_PRESETS:
            raise ConfigError(f"Unknown preset '{name}'. Choose from {list(ICHIMOKU_PRESETS)} or 'dynamic'")
        return cls(params=ICHIMOKU_PRESETS[name], **overrides)

    @property
    def label(self) -> str:
        if self.dynamic:
            return "Ichimoku dynamic (" + " | ".join(ICHIMOKU_PRESETS[p].label for p in self.regime_presets) + ")"
        return f"Ichimoku {self.params.label}"


# --------------------------------------------------------------------------------------
# Market microstructure, risk, backtest & live settings
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CostConfig:
    """Exchange fees and execution-cost model (see risk.CostModel for the maths)."""

    maker_fee: float = 0.001          # 0.10 % (Binance spot, VIP 0, no BNB discount)
    taker_fee: float = 0.001          # 0.10 %
    half_spread_bps: float = 1.0      # half the quoted bid/ask spread, in basis points
    slippage_bps: float = 2.0         # latency / queue-position slippage, in basis points
    impact_coefficient: float = 0.5   # Y in the square-root impact law  I = Y * sigma * sqrt(Q/V)
    use_market_impact: bool = True
    borrow_rate_annual: float = 0.05  # financing cost on short positions (margin interest / funding)

    def __post_init__(self) -> None:
        if min(self.maker_fee, self.taker_fee, self.half_spread_bps, self.slippage_bps,
               self.impact_coefficient, self.borrow_rate_annual) < 0:
            raise ConfigError("Cost parameters must be non-negative")


@dataclass(frozen=True)
class RiskConfig:
    """Position sizing, protective stops and portfolio-level circuit breakers."""

    initial_capital: float = 10_000.0
    sizing_method: SizingMethod = SizingMethod.FIXED_FRACTIONAL
    risk_per_trade: float = 0.01          # fixed-fractional: 1 % of equity at risk per trade
    kelly_multiplier: float = 0.5         # half-Kelly
    kelly_lookback_trades: int = 30
    kelly_min_trades: int = 10
    max_risk_per_trade: float = 0.03      # hard cap on the Kelly risk fraction
    max_position_pct: float = 1.0         # max notional / equity (1.0 = unlevered spot)
    min_order_notional: float = 10.0      # exchange minimum order value (quote currency)
    stop_method: StopMethod = StopMethod.ATR
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 4.0        # 2:1 reward-to-risk with the default stop
    cloud_buffer_atr: float = 0.5         # buffer beyond the Kumo boundary for cloud stops
    cloud_max_stop_atr: float = 5.0       # cap on the cloud-stop distance, in ATRs
    reward_risk_ratio: float = 2.0        # take-profit multiple of the initial risk for cloud stops
    use_take_profit: bool = True
    trailing_method: TrailingMethod = TrailingMethod.NONE
    trailing_atr_multiplier: float = 3.0
    max_drawdown_limit: float = 0.20      # circuit breaker: flatten + halt at a 20 % drawdown
    circuit_breaker_cooldown_bars: int = 168
    daily_loss_limit: float = 0.05        # halt for the rest of the UTC day after a 5 % loss

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ConfigError("initial_capital must be positive")
        if not 0 < self.risk_per_trade < 1 or not 0 < self.max_risk_per_trade < 1:
            raise ConfigError("risk fractions must be in (0, 1)")
        if self.max_position_pct <= 0:
            raise ConfigError("max_position_pct must be positive")
        if self.atr_sl_multiplier <= 0 or self.atr_tp_multiplier <= 0 or self.reward_risk_ratio <= 0:
            raise ConfigError("stop / target multipliers must be positive")
        if not 0 <= self.max_drawdown_limit < 1 or not 0 <= self.daily_loss_limit < 1:
            raise ConfigError("drawdown limits must be in [0, 1)")


@dataclass(frozen=True)
class BacktestConfig:
    risk_free_rate: float = 0.0           # annual; e.g. 0.04 for a USDT savings yield
    report_dir: Path = REPORT_DIR
    train_fraction: float = 0.7           # in-sample share used by the optimiser
    benchmark_label: str = "Buy & Hold"


@dataclass(frozen=True)
class ExchangeConfig:
    """Exchange connectivity. Secrets are excluded from repr() and serialisation."""

    exchange_id: str = field(default_factory=lambda: _env_str("EXCHANGE_ID", "binance"))
    api_key: str = field(default_factory=lambda: _env_str("EXCHANGE_API_KEY", ""), repr=False)
    api_secret: str = field(default_factory=lambda: _env_str("EXCHANGE_API_SECRET", ""), repr=False)
    use_testnet: bool = field(default_factory=lambda: _env_bool("EXCHANGE_USE_TESTNET", True))
    market_type: str = "spot"             # "spot" | "future" (USD-M perpetuals)
    request_timeout_ms: int = 30_000
    max_retries: int = 5

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def ccxt_options(self, authenticated: bool = False) -> dict[str, Any]:
        """Constructor options for a CCXT exchange instance."""
        options: dict[str, Any] = {
            "enableRateLimit": True,  # CCXT's built-in token-bucket throttle
            "timeout": self.request_timeout_ms,
            "options": {"defaultType": self.market_type},
        }
        if authenticated:
            options["apiKey"] = self.api_key
            options["secret"] = self.api_secret
        return options


@dataclass(frozen=True)
class DataConfig:
    symbol: str = field(default_factory=lambda: _env_str("SYMBOL", "BTC/USDT"))
    timeframe: str = field(default_factory=lambda: _env_str("TIMEFRAME", "1h"))
    history_days: int = 365
    source: DataSource = DataSource.EXCHANGE
    csv_path: Path | None = None
    db_path: Path = MARKET_DB_PATH
    synthetic_seed: int = 42
    min_bars: int = 300

    def __post_init__(self) -> None:
        timeframe_to_minutes(self.timeframe)  # validates
        if self.history_days <= 0:
            raise ConfigError("history_days must be positive")


@dataclass(frozen=True)
class LiveConfig:
    mode: TradingMode = field(
        default_factory=lambda: TradingMode(_env_str("TRADING_MODE", TradingMode.PAPER_TRADING.value))
    )
    feed: FeedType = FeedType.CCXT_PRO
    warmup_bars: int = 400
    bar_buffer: int = 1500                # closed bars kept in memory for indicator computation
    order_book_depth: int = 20
    poll_interval_sec: float = 5.0        # REST fallback polling period
    heartbeat_sec: float = 30.0
    stale_feed_timeout_sec: float = 180.0
    max_order_notional: float = 1_000.0   # hard per-order cap in LIVE mode (fat-finger guard)
    flatten_on_exit: bool = False
    replay_bars: int = 500                # bars replayed by the offline ReplayFeed
    replay_speed: float = 0.0             # seconds between replayed bars (0 = as fast as possible)
    journal_db_path: Path = JOURNAL_DB_PATH


@dataclass(frozen=True)
class AppConfig:
    """Top-level, immutable configuration object passed through the whole system."""

    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    @property
    def periods_per_year(self) -> float:
        return periods_per_year(self.data.timeframe)

    def with_updates(self, **sections: Any) -> AppConfig:
        """Return a copy with whole sections replaced, e.g. ``cfg.with_updates(risk=new_risk)``."""
        return replace(self, **sections)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dictionary of the configuration, with secrets removed."""
        return _jsonable(self)


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        # Fields declared with repr=False (API keys) are never serialised.
        return {f.name: _jsonable(getattr(obj, f.name)) for f in fields(obj) if f.repr}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return obj


def load_config() -> AppConfig:
    """Build the default configuration (environment variables already applied)."""
    return AppConfig()


# --------------------------------------------------------------------------------------
# Structured logging
# --------------------------------------------------------------------------------------
_STANDARD_RECORD_KEYS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line: machine-parseable logs for audit and post-mortems.

    Anything passed through ``extra={...}`` becomes a top-level JSON field, e.g.
    ``logger.info("order filled", extra={"event": "fill", "price": 64000.5})``.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_RECORD_KEYS})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_HANDLER_TAG = "_ichimoku_bot_handler"


def setup_logging(level: str = "INFO", log_dir: Path = LOG_DIR, json_file: bool = True) -> None:
    """Configure console (human-readable) and rotating JSON-lines file logging.

    Idempotent: calling it again replaces the handlers it installed previously.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s", "%H:%M:%S"))
    setattr(console, _HANDLER_TAG, True)
    root.addHandler(console)

    if json_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "trading_bot.jsonl", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(JsonFormatter())
        setattr(file_handler, _HANDLER_TAG, True)
        root.addHandler(file_handler)

    for noisy in ("matplotlib", "PIL", "ccxt", "asyncio", "urllib3", "aiohttp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
