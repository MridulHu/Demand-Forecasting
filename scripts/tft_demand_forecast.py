import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

import lightning.pytorch as pl

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from lightning.pytorch.callbacks import EarlyStopping


OUTPUT_DIR = "outputs/plots/tft_forecasting"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():

    # -----------------------------
    # Load data
    # -----------------------------
    data = pd.read_csv("data/retail_store_inventory.csv")
    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values(["Store ID", "Product ID", "Date"])

    # create time index
    data["time_idx"] = (
        data.groupby(["Store ID", "Product ID"])
        .cumcount()
    )

    # time features
    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["Month"] = data["Date"].dt.month


    # -----------------------------
    # Lag features (important for forecasting)
    # -----------------------------

    data["lag_1"] = data.groupby(["Store ID","Product ID"])["Units Sold"].shift(1)
    data["lag_7"] = data.groupby(["Store ID","Product ID"])["Units Sold"].shift(7)
    data["lag_14"] = data.groupby(["Store ID","Product ID"])["Units Sold"].shift(14)

    # remove rows with missing lag values
    data = data.dropna().reset_index(drop=True)


    # -----------------------------
    # Dataset configuration
    # -----------------------------
    max_encoder_length = 180
    max_prediction_length = 30

    training_cutoff = data["time_idx"].max() - max_prediction_length


    # -----------------------------
    # Training dataset
    # -----------------------------
    training = TimeSeriesDataSet(
        data[data.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="Units Sold",
        group_ids=["Store ID", "Product ID"],

        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,

        static_categoricals=["Store ID", "Product ID"],

        time_varying_known_reals=[
            "time_idx",
            "Price",
            "Discount",
            "Competitor Pricing",
            "DayOfWeek",
            "Month",
            "lag_1",
            "lag_7",
            "lag_14"
        ],

        time_varying_unknown_reals=["Units Sold"],

        target_normalizer=GroupNormalizer(
            groups=["Store ID", "Product ID"]
        ),
    )


    # -----------------------------
    # Validation dataset
    # -----------------------------
    validation = TimeSeriesDataSet.from_dataset(
        training,
        data,
        predict=True,
        stop_randomization=True
    )


    # -----------------------------
    # DataLoaders
    # -----------------------------
    train_loader = training.to_dataloader(
        train=True,
        batch_size=128,
        num_workers=8,
        persistent_workers=True
    )

    val_loader = validation.to_dataloader(
        train=False,
        batch_size=128,
        num_workers=8,
        persistent_workers=True
    )


    # -----------------------------
    # Model
    # -----------------------------
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.003,
        hidden_size=128,
        attention_head_size=8,
        dropout=0.1,
        hidden_continuous_size=64,
        loss=QuantileLoss(),
    )


    # -----------------------------
    # Early stopping
    # -----------------------------
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=0.0001,
        mode="min"
    )


    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=150,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        log_every_n_steps=150
    )


    # -----------------------------
    # Train
    # -----------------------------
    trainer.fit(
        tft,
        train_loader,
        val_loader
    )


    # -----------------------------
    # Prediction (full timeline)
    # -----------------------------

    predictions = tft.predict(
    validation,
    mode="raw",
    return_x=True
    )

    raw_predictions = predictions.output
    x = predictions.x

    pred = raw_predictions["prediction"].cpu().numpy()

    # take median quantile prediction
    pred = pred[:, :, 0]

    # flatten prediction timeline
    pred_values = pred.flatten()

    # construct time index for prediction window
    pred_time = []

    for i in range(pred.shape[0]):
        base = x["decoder_time_idx"][i].cpu().numpy()
        pred_time.extend(base)

    pred_df = pd.DataFrame({
    "time_idx": pred_time,
    "prediction": pred_values
    })

    # Average overlapping predictions
    pred_df = pred_df.groupby("time_idx")["prediction"].mean().reset_index()

    # attach store/product ids from original data
    pred_df = pd.merge(
        pred_df,
        data[["time_idx","Store ID","Product ID"]],
        on="time_idx",
        how="left"
    )

    # -----------------------------
# Merge predictions with data
    # -----------------------------

    merged = pd.merge(
        data,
        pred_df,
        on=["time_idx","Store ID","Product ID"],
        how="left"
    )

    # -----------------------------
    # Plot example product
    # -----------------------------

    example_store = merged["Store ID"].iloc[0]
    example_product = merged["Product ID"].iloc[0]

    merged = merged[
        (merged["Store ID"] == example_store) &
        (merged["Product ID"] == example_product)
    ]

    # smoothing for presentation
    merged["prediction"] = merged["prediction"].rolling(
        3,
        min_periods=1
    ).mean()

   # -----------------------------
    # Future forecast
    # -----------------------------

    future_values = pred[-1]
    future_steps = len(future_values)

    last_idx = merged["time_idx"].max()

    future_index = np.arange(last_idx + 1, last_idx + future_steps + 1)

    future_df = pd.DataFrame({
        "time_idx": future_index,
        "prediction": future_values
    })

    merged = pd.concat([merged, future_df], ignore_index=True)
    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(14,5))

    plt.plot(
        merged["Units Sold"],
        label="Actual Demand",
        alpha=0.8
    )

    plt.plot(
        merged["prediction"],
        label="TFT Forecast",
        linewidth=3,
        color="orange"
    )

    split = merged["Units Sold"].last_valid_index()

    plt.axvline(
        x=split,
        linestyle="--",
        color="black",
        linewidth=2,
        label="Forecast Start"
    )

    plt.title("Temporal Fusion Transformer Demand Forecast")
    plt.xlabel("Time")
    plt.ylabel("Units Sold")

    plt.legend()

    file = f"{OUTPUT_DIR}/tft_full_prediction.png"

    plt.savefig(file, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved →", file)


# Windows multiprocessing fix
if __name__ == "__main__":
    main()