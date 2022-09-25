import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import skorch
import torch
from sklearn.metrics import classification_report
from skorch import NeuralNet
from skorch.helper import predefined_split

import wandb
from dl4nlp.dataset import DATASET_DIR, DataModule
from dl4nlp.features_extractor import UNICODE_CATEGORIES, Feature, FeaturesExtractor
from dl4nlp.models import probe
from dl4nlp.models.bert import EMBEDDINGS_DIR, MBERT_MODEL
from dl4nlp.utils import (
    DEVICE,
    EnumAction,
    Optimizer,
    WandbLogger,
    pandas_set_option,
    seed_everything,
    skorch_accuracy,
)


def train_eval(args, wandb_run):
    print(f"Using device '{DEVICE}'")
    print("Loading datamodule...")
    data_module = DataModule(
        data_dir=args.data_dir,
        embeddings_dir=args.embeddings_dir,
        bert_batch_size=args.bert_batch_size,
        model_name=args.bert_model_name,
    )

    print("Extracting features...")
    features_extractor = FeaturesExtractor(
        features=args.features,
        unicode_categories=args.unicode_categories,
    )
    data_module.init_features_embeddings(features_extractor)

    net_kwargs = {}
    if args.optimizer is Optimizer.SGD:
        net_kwargs["optimizer__momentum"] = args.momentum

    net = NeuralNet(
        module=probe.ClassifierHead,
        module__input_dim=data_module.num_features,
        module__output_dim=data_module.num_classes,
        module__hidden_dims=args.hidden_dims,
        criterion=torch.nn.CrossEntropyLoss,
        train_split=predefined_split(data_module.dev),
        max_epochs=1000,
        device=DEVICE,
        verbose=1,
        lr=args.learning_rate,
        optimizer=args.optimizer.value,
        iterator_train__num_workers=min(os.cpu_count(), 8),
        iterator_valid__num_workers=min(os.cpu_count(), 8),
        iterator_train__pin_memory=True,
        iterator_valid__pin_memory=True,
        batch_size=args.batch_size,
        callbacks=[
            skorch.callbacks.EpochScoring(
                skorch_accuracy, lower_is_better=False, name="valid_accuracy"
            ),
            skorch.callbacks.EarlyStopping(),
            skorch.callbacks.ProgressBar(),
            WandbLogger(wandb_run),
        ],
        **net_kwargs,
    )

    print("Training model...")
    net.fit(data_module.train)

    print("Evaluating...")
    predictions = net.predict(data_module.test)
    pred_y = np.argmax(predictions, axis=-1)
    true_y = data_module.test.targets
    eval_report = pd.DataFrame(
        classification_report(
            true_y,
            pred_y,
            target_names=data_module.classes,
            zero_division=0,
            output_dict=True,
        )
    ).T

    wandb_run.log({"test/accuracy": eval_report.precision.accuracy})
    wandb_run.log(
        {
            "test/classification_report": eval_report.reset_index().rename(
                columns={"index": "lang"}
            )
        }
    )
    with pandas_set_option("display.precision", 3), pandas_set_option(
        "display.max_rows", None
    ):
        print(eval_report)

    return net


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--experiment-name",
        "-n",
        type=str,
        default=None,
        help="Name of the experiment to log into WandB",
    )
    group = parser.add_argument_group("Data")
    group.add_argument(
        "--data-dir",
        type=Path,
        default=DATASET_DIR,
        help="Path to the dataset directory",
    )
    group.add_argument(
        "--embeddings-dir",
        type=Path,
        default=EMBEDDINGS_DIR,
        help="Path to the embeddings' directory",
    )

    group = parser.add_argument_group("Features extraction")
    group.add_argument(
        "--features",
        nargs="+",
        type=Feature,
        action=EnumAction,
        default=[],
        help="Features to include in classification process",
    )
    group.add_argument(
        "--unicode-categories",
        nargs="+",
        type=str,
        default=UNICODE_CATEGORIES,
        help="Unicode categories to include in the features",
    )

    group = parser.add_argument_group("BERT model")
    group.add_argument(
        "--bert-batch-size", type=int, default=4, help="Batch size for BERT model"
    )
    group.add_argument(
        "--bert-model-name",
        type=str,
        default=MBERT_MODEL,
        help="Name of pretrained BERT model to use",
    )

    group = parser.add_argument_group("Classification model")
    group.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[],
        help="Dimensions of hidden layers for classification head",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for classification model",
    )
    group.add_argument(
        "--optimizer",
        type=Optimizer,
        action=EnumAction,
        default=Optimizer.SGD,
        help="Training optimizer",
    )
    group.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=0.2,
        help="Optimizer's learning rate",
    )
    group.add_argument(
        "--momentum", type=float, default=0.9, help="Optimizer's momentum"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run = wandb.init(name=args.experiment_name, project="dl4nlp")
    wandb.config.update(args)
    seed_everything(13331)
    train_eval(args, run)