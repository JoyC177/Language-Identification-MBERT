from pathlib import Path
from typing import List, Union

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .models.bert import (
    MBERT_MODEL,
    NUM_LAYERS,
    XLMBERT_MODEL,
    generate_bert_embeddings,
    load_model,
)

DATASET_DIR = "data/wili-2018-split/"
EMBEDDINGS_DIR = "embeddings/"
TRAIN_FNAME = "train.csv"
DEV_FNAME = "dev.csv"
TEST_FNAME = "test.csv"
LABELS_FNAME = "labels.csv"
EMBEDDINGS_FNAME = "{}_embeddings_{}.h5"


class DataModule:
    def __init__(
        self,
        data_dir: str = DATASET_DIR,
        embeddings_dir: str = EMBEDDINGS_DIR,
        bert_batch_size: int = 32,
        save_embeddings: bool = True,
        model_name: str = MBERT_MODEL,
        embeddings_layer: int = NUM_LAYERS,
    ):
        self.data_dir = Path(data_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.bert_batch_size = bert_batch_size
        self.save_embeddings = save_embeddings
        self.model_name = model_name
        self.embeddings_layer = embeddings_layer
        self.bert_embeddings = {}
        self.features_embeddings = {}

        self._init_datasets()
        self._init_bert_embeddings()

    @property
    def num_features(self):
        num = self.bert_embeddings["train"].shape[-1]
        features = self.features_embeddings.get("train")
        if features is not None:
            num += features.shape[-1]
        return num

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def train(self):
        return EmbeddingsDataset(
            self.bert_embeddings["train"],
            self.train_ds.targets,
            self.features_embeddings.get("train"),
        )

    @property
    def dev(self):
        return EmbeddingsDataset(
            self.bert_embeddings["dev"],
            self.dev_ds.targets,
            self.features_embeddings.get("dev"),
        )

    @property
    def test(self):
        return EmbeddingsDataset(
            self.bert_embeddings["test"],
            self.test_ds.targets,
            self.features_embeddings.get("test"),
        )

    def init_features_embeddings(self, features_extractor):
        self.features_embeddings = {}
        for split_name, dataset in self._dataset_splits.items():
            embeddings = features_extractor(dataset.sentences)
            self.features_embeddings[split_name] = embeddings

    def _init_datasets(self):
        classes = pd.read_csv(
            self.data_dir / LABELS_FNAME, na_filter=False, usecols=["lang"]
        )
        self.classes = list(classes.lang)
        self.train_ds = WiliDataset(self.data_dir / TRAIN_FNAME, self.classes)
        self.dev_ds = WiliDataset(self.data_dir / DEV_FNAME, self.classes)
        self.test_ds = WiliDataset(self.data_dir / TEST_FNAME, self.classes)
        self._dataset_splits = {
            "train": self.train_ds,
            "dev": self.dev_ds,
            "test": self.test_ds,
        }

    def _init_bert_embeddings(self):
        self.bert_model = None
        self.bert_tokenizer = None
        for split_name, dataset in self._dataset_splits.items():
            embeddings = self._get_bert_embeddings(split_name, dataset)
            self.bert_embeddings[split_name] = embeddings

        del self.bert_model
        del self.bert_tokenizer

    def _get_bert_embeddings(self, split_name: str, dataset: "WiliDataset"):
        embeddings_file = self.embeddings_dir / EMBEDDINGS_FNAME.format(
            self.model_name, split_name
        )

        if embeddings_file.exists():
            with h5py.File(embeddings_file, "r") as h5f:
                embeddings = torch.from_numpy(h5f[f"l{self.embeddings_layer}"][:])
            print(f"Loaded BERT embeddings from '{embeddings_file.absolute()}'")
        else:
            print(
                f"BERT embeddings don't exist in path '{embeddings_file.absolute()}'. "
                "Creating new..."
            )
            if self.bert_model is None:
                print("Initializing BERT model...")
                self.bert_model, self.bert_tokenizer = load_model(self.model_name)

            print("Generating BERT embeddings...")
            embeddings = generate_bert_embeddings(
                self.bert_model, self.bert_tokenizer, dataset, self.bert_batch_size
            )

            if self.save_embeddings:
                embeddings_file.parent.mkdir(exist_ok=True, parents=True)
                with h5py.File(embeddings_file, "w") as h5f:
                    for l in range(len(embeddings)):
                        layer = embeddings[l].cpu()
                        h5f.create_dataset(f"l{l}", data=layer)
                print(f"BERT embeddings saved to '{embeddings_file.absolute()}'")

        return embeddings


class EmbeddingsDataset(Dataset):
    def __init__(self, bert_embeddings, targets, features_embeddings=None):
        self.bert_embeddings = bert_embeddings
        self.features_embeddings = features_embeddings
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        embeddings = [self.bert_embeddings[idx]]
        if self.features_embeddings is not None:
            embeddings.append(self.features_embeddings[idx])
        return torch.cat(embeddings), self.targets[idx]


class WiliDataset(Dataset):
    def __init__(self, dataset_path: Union[str, Path], classes: List[str]):
        self.data = pd.read_csv(
            dataset_path, dtype="str", na_filter=False, usecols=["sentence", "lang"]
        )
        self.classes = classes
        self.label2idx = {l: i for i, l in enumerate(classes)}

    @property
    def sentences(self):
        return self.data.sentence

    @property
    def targets(self):
        return [self.label2idx[l] for l in self.data.lang]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, lang = self.data.iloc[idx]
        return sentence, self.label2idx[lang]
