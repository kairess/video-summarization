import argparse
import json
import os

import pytorch_lightning as pl
from datamodule import DescriptionDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader

from v2021 import DescriptionModel


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
    parser.add_argument("-bs", "--batch-size", default=1, help="Batch size")
    parser.add_argument("-w", "--weights", default=None, help="Pretrained weights")
    args = parser.parse_args()

    # Load dataset.
    train_videos, validate_videos, test_videos = load_splits(args.splits_file)

    train_set = DescriptionDataset(
        train_videos, args.data_directory, args.video_features_file
    )
    validate_set = DescriptionDataset(
        validate_videos, args.data_directory, args.video_features_file
    )
    test_set = DescriptionDataset(
        test_videos, args.data_directory, args.video_features_file
    )
    cpu_count = os.cpu_count()
    bs = int(args.batch_size)
    train_dataloader = DataLoader(train_set, batch_size=bs, num_workers=0)
    validate_dataloader = DataLoader(validate_set, batch_size=bs, num_workers=0)
    test_dataloader = DataLoader(test_set, batch_size=bs, num_workers=0)

    # Load model.
    if args.weights:
        model = DescriptionModel.load_from_checkpoint(
            args.weights, individual_logs="description_individual_logs.txt"
        )
    else:
        model = DescriptionModel()

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
            version="description",
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
        with open(f"description_individual_logs_{model.global_rank}.txt", "w") as wf:
            for key, items in model.tta_logs.items():
                tp, fp, fn = 0, 0, 0
                for itp, ifp, ifn in items:
                    tp += itp
                    fp += ifp
                    fn += ifn
                item_len = len(model.tta_logs[key])
                precision = 0
                recall = 0
                f1 = 0
                log_string = (
                    f"{key},{tp},{fp},{fn},{precision:.2f},{recall:.2f},{f1:.2f}\n"
                )
                wf.write(log_string)
