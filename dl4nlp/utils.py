import argparse
import enum
import os
import random
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import skorch
import torch
from sklearn.metrics import accuracy_score
from skorch.callbacks.logging import filter_log_keys
from tqdm.auto import tqdm
from joblib import Parallel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def skorch_accuracy(net, ds, y=None):
    y_true = [y for _, y in ds]
    y_pred = np.argmax(net.predict(ds), axis=-1)
    return accuracy_score(y_true, y_pred)


class WandbLogger(skorch.callbacks.WandbLogger):
    def on_epoch_end(self, net, **kwargs):
        """Log values from the last history step and save best model"""
        hist = net.history[-1]
        keys_kept = filter_log_keys(hist, keys_ignored=self.keys_ignored_)
        logged_vals = {k.replace("_", "/", 1): hist[k] for k in keys_kept}
        self.wandb_run.log(logged_vals)

        # save best model
        if self.save_model and hist["valid_loss_best"]:
            model_path = Path(self.wandb_run.dir) / "best_model.pth"
            with model_path.open("wb") as model_file:
                net.save_params(f_params=model_file)


class Optimizer(Enum):
    SGD = torch.optim.SGD
    Adam = torch.optim.Adam
    AdamW = torch.optim.AdamW


class Activation(Enum):
    ReLU = torch.nn.ReLU
    Sigmoid = torch.nn.Sigmoid
    Tanh = torch.nn.Tanh


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.name for e in enum_type))
        super(EnumAction, self).__init__(**kwargs)
        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        if isinstance(values, list):
            value = [self._enum[v] for v in values]
        else:
            value = self._enum[values]
        setattr(namespace, self.dest, value)


@contextmanager
def pandas_set_option(option: str, value):
    old_value = pd.get_option(option)
    pd.set_option(option, value)
    yield
    pd.set_option(option, old_value)
