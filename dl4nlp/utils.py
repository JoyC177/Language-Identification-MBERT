import os
import random
from pathlib import Path

import numpy as np
import skorch
import torch
from sklearn.metrics import accuracy_score
from skorch.callbacks.logging import filter_log_keys

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
