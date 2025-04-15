from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting import TimeSeriesDataSet
import requests
import pandas as pd


def fetch_data(start_date, end_date=None):
    url = f"https://api.frankfurter.dev/v1/{start_date}..{end_date}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(
            f"API request failed with status code {response.status_code}")

    data = response.json()

    records = []
    for date, rates in data['rates'].items():
        for currency, rate in rates.items():
            records.append({
                'date': date,
                'currency': currency,
                'rate': rate
            })

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])

    return df


def preprocess_data(df, target_currency="USD"):
    if target_currency:
        df = df[df['currency'] == target_currency].copy()

    df = df.sort_values('date')

    df['time_idx'] = (df['date'] - df['date'].min()).dt.days

    df['day_of_week'] = df['date'].dt.dayofweek.astype(str)
    df['month'] = df['date'].dt.month.astype(str)

    for i in range(1, 8):
        df[f'lag_{i}'] = df['rate'].shift(i)

    df = df.dropna()

    df['group_id'] = 0

    return df


def create_datasets(df, max_prediction_length=30, max_encoder_length=90):
    training_cutoff = df['time_idx'].max() - max_prediction_length

    training = TimeSeriesDataSet(
        data=df[df['time_idx'] <= training_cutoff],
        time_idx="time_idx",
        target="rate",
        group_ids=["group_id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        time_varying_known_categoricals=["day_of_week", "month"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "rate",
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_4",
            "lag_5",
            "lag_6",
            "lag_7",
        ],
        target_normalizer=GroupNormalizer(
            groups=["group_id"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        min_prediction_idx=training_cutoff + 1,
        stop_randomization=True
    )

    return training, validation


def create_dataloaders(training, validation, batch_size=8):
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
    )

    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
    )

    return train_dataloader, val_dataloader
