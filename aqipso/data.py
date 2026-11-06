"""Data loading, cleaning, feature engineering and chronological splitting.

Sources: the CPCB daily AQI bulletins for Delhi (the target and its history) and
ERA5 reanalysis weather for central Delhi. Everything is causal: a feature for
day t uses only information available by the end of day t, and the target is the
officially reported AQI of day t + 1.
"""
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg

WEATHER_RENAME = {
    "temperature_2m_max": "temp_max", "temperature_2m_min": "temp_min",
    "temperature_2m_mean": "temp_mean", "precipitation_sum": "rain",
    "relative_humidity_2m_mean": "humidity", "wind_speed_10m_max": "wind_max",
    "wind_speed_10m_mean": "wind_mean", "wind_direction_10m_dominant": "wind_dir",
    "surface_pressure_mean": "pressure", "shortwave_radiation_sum": "radiation",
    "dew_point_2m_mean": "dew_point",
}


@dataclass
class Dataset:
    """Chronologically split model-ready data."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    raw: pd.DataFrame
    report: dict = field(default_factory=dict)


def load_aqi(city: str = cfg.CITY, path=cfg.AQI_PATH) -> pd.Series:
    """Daily bulletin AQI for one city, on a complete daily calendar (gaps are NaN)."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["city"] == city].drop_duplicates("date").set_index("date").sort_index()
    if df.empty:
        raise ValueError(f"No bulletin records for {city!r}")
    return df["aqi"].astype(float).asfreq("D").rename("AQI")


def load_weather(weather_path=cfg.WEATHER_PATH) -> pd.DataFrame:
    """Daily ERA5 reanalysis weather for central Delhi."""
    daily = pd.DataFrame(json.loads(open(weather_path).read())["daily"])
    w = daily.assign(time=pd.to_datetime(daily["time"])).set_index("time").rename(columns=WEATHER_RENAME)
    # Wind direction is circular, so 359 degrees and 1 degree must be neighbours.
    radians = np.radians(w.pop("wind_dir"))
    w["wind_dir_sin"], w["wind_dir_cos"] = np.sin(radians), np.cos(radians)
    return w


def load_raw(city: str = cfg.CITY) -> pd.DataFrame:
    aqi = load_aqi(city)
    return aqi.to_frame().join(load_weather(), how="left")


def clean(df: pd.DataFrame, test_start: str = cfg.TEST_START):
    """Fill gaps without looking into the future.

    Forward fill copies the last known value, which was genuinely available at
    prediction time; anything still missing at the start gets training medians.
    The observed AQI is kept separately so the target is never an imputed value.
    """
    observed_aqi = df["AQI"].copy()
    out = df.ffill()
    out = out.fillna(df[df.index < test_start].median())
    return out, observed_aqi


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add AQI history, weather change and seasonal features, all known at day t."""
    out = df.copy()
    aqi = df["AQI"]
    for lag in (1, 2, 3, 7):
        out[f"AQI_lag_{lag}"] = aqi.shift(lag)
    for window in (3, 7, 14, 30):
        out[f"AQI_roll_mean_{window}"] = aqi.rolling(window).mean()
    out["AQI_roll_std_7"] = aqi.rolling(7).std()
    out["AQI_change_1"] = aqi.diff()
    for col in ("temp_min", "pressure", "dew_point", "wind_mean"):
        out[f"{col}_change_1"] = df[col].diff()
    out["rain_3d"] = df["rain"].rolling(3).sum()

    # The target is t + 1, so the calendar describes the day being forecast.
    target_day = out.index + pd.Timedelta(days=1)
    out["doy_sin"] = np.sin(2 * np.pi * target_day.day_of_year / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * target_day.day_of_year / 365.25)
    out["dow_sin"] = np.sin(2 * np.pi * target_day.day_of_week / 7)
    out["dow_cos"] = np.cos(2 * np.pi * target_day.day_of_week / 7)
    return out


def build_dataset(city: str = cfg.CITY, test_start: str = cfg.TEST_START) -> Dataset:
    """Run the full pipeline from raw files to train/test matrices."""
    raw = load_raw(city)
    cleaned, observed_aqi = clean(raw, test_start)
    features = engineer_features(cleaned)

    y = observed_aqi.shift(-1).rename("AQI_next_day")
    data = features.assign(AQI_next_day=y).dropna()
    X, y = data.drop(columns="AQI_next_day"), data["AQI_next_day"]

    is_train = X.index < test_start
    report = {
        "city": city,
        "raw_days": len(raw),
        "date_range": [str(raw.index.min().date()), str(raw.index.max().date())],
        "missing_pct": (raw.isna().mean() * 100).round(2).to_dict(),
        "aqi_missing_days": int(raw["AQI"].isna().sum()),
        "model_samples": len(X),
        "train_samples": int(is_train.sum()),
        "test_samples": int((~is_train).sum()),
        "train_range": [str(X.index[is_train].min().date()), str(X.index[is_train].max().date())],
        "test_range": [str(X.index[~is_train].min().date()), str(X.index[~is_train].max().date())],
        "n_features": X.shape[1],
    }
    return Dataset(X[is_train], y[is_train], X[~is_train], y[~is_train], raw, report)
