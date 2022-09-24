import numpy as np
import skorch
import torch
from sklearn.metrics import classification_report
from skorch import NeuralNet
from skorch.helper import predefined_split

import wandb
from dl4nlp.dataset import DataModule
from dl4nlp.features_extractor import Feature, FeaturesExtractor
from dl4nlp.models import probe
from dl4nlp.utils import DEVICE, WandbLogger, seed_everything, skorch_accuracy


def train_eval(wandb_run):
    print(f"Using device '{DEVICE}'")
    print("Loading datamodule...")
    data_module = DataModule(bert_batch_size=4)

    print("Extracting features...")
    features_extractor = FeaturesExtractor(
        features=[
            Feature.UNICODE_CATEGORY,
        ]
    )
    data_module.init_features_embeddings(features_extractor)

    net = NeuralNet(
        module=probe.ClassifierHead,
        module__input_dim=data_module.num_features,
        module__output_dim=data_module.num_classes,
        criterion=torch.nn.CrossEntropyLoss,
        train_split=predefined_split(data_module.dev),
        max_epochs=1000,
        device=DEVICE,
        verbose=1,
        lr=0.2,
        optimizer=torch.optim.SGD,
        optimizer__momentum=0.9,
        iterator_train__num_workers=min(os.cpu_count(), 8),
        iterator_valid__num_workers=min(os.cpu_count(), 8),
        iterator_train__pin_memory=True,
        iterator_valid__pin_memory=True,
        batch_size=1024,
        callbacks=[
            skorch.callbacks.EpochScoring(
                skorch_accuracy, lower_is_better=False, name="valid_accuracy"
            ),
            skorch.callbacks.EarlyStopping(),
            skorch.callbacks.ProgressBar(),
            WandbLogger(wandb_run),
        ],
    )

    print("Training model...")
    net.fit(data_module.train)

    print("Evaluating...")
    predictions = net.predict(data_module.test)
    pred_y = np.argmax(predictions, axis=-1)
    true_y = data_module.test.targets
    eval_report = classification_report(
        true_y, pred_y, target_names=data_module.classes, zero_division=0
    )
    print(eval_report)

    return net


if __name__ == "__main__":
    run = wandb.init(project="dl4nlp")
    seed_everything(13331)
    train_eval(run)
