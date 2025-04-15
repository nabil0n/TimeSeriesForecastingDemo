from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning.pytorch as pl
import matplotlib.pyplot as plt


def build_model(training):
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    return model


def train_model(model, train_dataloader, val_dataloader, max_epochs=30):
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        verbose=False,
        mode="min"
    )

    lr_logger = LearningRateMonitor()

    logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        precision=32,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, lr_logger],
        logger=logger,
    )

    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
    )

    return trainer


def make_predictions(model, val_dataloader):
    raw_predictions, x, y, idx, decoder_lengths = model.predict(
        val_dataloader,
        return_x=True,
        return_y=True,
        mode="quantiles",
    )

    if raw_predictions.shape[1] == 30 and raw_predictions.shape[2] == 7:
        raw_predictions = raw_predictions.transpose(1, 2)
        print("Transposed predictions to correct dimensions")

    return raw_predictions, x, y, idx, decoder_lengths


def visualize_predictions(model, raw_predictions, x):
    print(f"Raw predictions shape: {raw_predictions.shape}")

    if len(raw_predictions.shape) == 2:
        print("Note: Predictions are 2D - this suggests single quantile output")
        print(
            f"Sample prediction values (first 3 days): {raw_predictions[0, :3].cpu().numpy()}")

        raw_predictions = raw_predictions.unsqueeze(1)
        print(f"Reshaped predictions: {raw_predictions.shape}")
    else:
        print("Sample prediction values (first 3 days):")
        for i in range(min(7, raw_predictions.shape[1])):
            print(f"Quantile {i}: {raw_predictions[0, i, :3].cpu().numpy()}")

    fig, ax = plt.subplots(figsize=(12, 8))

    try:
        model.plot_prediction(
            x, raw_predictions, idx=0, add_loss_to_title=True, ax=ax
        )
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 8))
        sample_idx = 0

        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'black']

        if len(raw_predictions.shape) == 3 and raw_predictions.shape[1] > 1:
            quantile_labels = [
                "10th", "20th", "50th (Median)", "80th", "90th", "Lower PI", "Upper PI"]

            for i in range(raw_predictions.shape[1]):
                label = quantile_labels[i] if i < len(
                    quantile_labels) else f"Quantile {i+1}"
                style = line_styles[i % len(line_styles)]
                color = colors[i % len(colors)]

                ax.plot(
                    raw_predictions[sample_idx, i].cpu().numpy(),
                    label=label,
                    linestyle=style,
                    color=color,
                    linewidth=2
                )
        else:
            ax.plot(
                raw_predictions[sample_idx, 0].cpu().numpy()
                if len(raw_predictions.shape) == 3
                else raw_predictions[sample_idx].cpu().numpy(),
                label="Forecast",
                color="blue",
                linewidth=2
            )

        ax.set_title("USD-EUR Exchange Rate 30-Day Forecast", fontsize=14)
        ax.set_xlabel("Days in Forecast Horizon", fontsize=12)
        ax.set_ylabel("USD to EUR Exchange Rate", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    plt.show()
