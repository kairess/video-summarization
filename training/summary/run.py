import argparse
import json
import os

import pytorch_lightning as pl
from datamodule import SummaryDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader

from v2021 import SummaryModel


def load_splits(splits_file):
    with open(splits_file, "r") as rf:
        splits = json.load(rf)
    return splits["train"], splits["validate"], splits["test"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-directory", help="Directory with .mp4 and .json files"
    )
    parser.add_argument(
        "-v", "--video-features-file", help=".h5 file with preprocessed video features"
    )
    parser.add_argument(
        "-s", "--splits-file", help=".json file with train/validate/test splits"
    )
    parser.add_argument(
        "-t", "--train", default=False, action="store_true", help="Run training"
    )
    parser.add_argument(
        "-q", "--quick", default=False, action="store_true", help="Do a quick test run"
    )
    parser.add_argument("-w", "--weights", default=None, help="Pretrained weights")
    args = parser.parse_args()

    # Load dataset.
    train_videos, validate_videos, test_videos = load_splits(args.splits_file)

    train_set = SummaryDataset(
        train_videos, args.data_directory, args.video_features_file
    )
    validate_set = SummaryDataset(
        validate_videos, args.data_directory, args.video_features_file
    )
    test_set = SummaryDataset(
        test_videos, args.data_directory, args.video_features_file
    )
    cpu_count = os.cpu_count()
    train_dataloader = DataLoader(train_set, batch_size=1, num_workers=cpu_count)
    validate_dataloader = DataLoader(validate_set, batch_size=1, num_workers=cpu_count)
    test_dataloader = DataLoader(test_set, batch_size=1, num_workers=cpu_count)

    # Load model.
    if args.weights:
        model = SummaryModel.load_from_checkpoint(
            args.weights, individual_logs="summary_individual_logs.txt"
        )
    else:
        model = SummaryModel()

    # Initialize trainer.
    gpus = 0 if args.quick else -1
    accelerator = None if args.quick else "ddp"
    plugins = None if args.quick else DDPPlugin(find_unused_parameters=False)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=15,
        verbose=True,
        mode="min",
        strict=True,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        save_top_k=15,
    )

    callbacks = [early_stop_callback, checkpoint_callback] if args.train else None
    logger = (
        TensorBoardLogger(
            save_dir=os.getcwd(),
            version="summary",
            name="lightning_logs",
        )
        if args.train
        else False
    )

    trainer = pl.Trainer(
        gpus=gpus,
        accelerator=accelerator,
        callbacks=callbacks,
        # max_epochs=15,
        plugins=plugins,
        log_every_n_steps=1,
        logger=logger,
    )

    # Run training.
    if args.train:
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=validate_dataloader,
        )

    # Run testing.
    else:
        model.eval()
        trainer.test(model, dataloaders=test_dataloader)
        with open(f"summary_individual_logs_{model.global_rank}.txt", "w") as wf:
            for key, items in model.tta_logs.items():
                precision, recall, f1 = 0, 0, 0
                for tp, fp, fn in items:
                    prec = tp / (tp + fp) if tp + fp else 0
                    rec = tp / (tp + fn) if tp + fn else 0
                    f = 2 * prec * rec / (prec + rec)
                    precision += prec
                    recall += rec
                    f1 += f
                item_len = len(model.tta_logs[key])
                precision /= item_len
                recall /= item_len
                f1 /= item_len
                log_string = (
                    f"{key},{tp},{fp},{fn},{precision:.2f},{recall:.2f},{f1:.2f}\n"
                )
                wf.write(log_string)
