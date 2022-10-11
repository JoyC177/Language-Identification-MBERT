import pickle
import unicodedata
from collections import Counter
from enum import Enum, auto
from typing import List, Optional, Union

import torch
from tqdm.auto import tqdm

UNICODE_CATEGORIES = ["Ll", "Zs", "Lu", "Po", "Pd", "Lo", "Mn", "Ps", "Pe", "Mc"]
LETTER_CATEGORIES = ["Lu", "Ll", "Lt", "Lm"]  # "Lo"


class Feature(Enum):
    UNICODE_CATEGORY = auto()
    LETTERS_COUNT = auto()


class FeaturesExtractor:
    def __init__(
        self,
        features: Optional[List[Union[Feature, str]]] = None,
        unicode_categories: List[str] = UNICODE_CATEGORIES,
        normalize=True,
    ):
        if features is None:
            features = []

        for i, feature in enumerate(features):
            if isinstance(feature, str):
                features[i] = Feature[feature]

        self.features = features
        self.unicode_categories = unicode_categories
        self.normalize = normalize

        if Feature.LETTERS_COUNT in self.features:
            with open("data/letters.pickle", "rb") as f:
                self.letters = pickle.load(f)

    def process_sentence(self, paragraph: str):
        features = [torch.empty(0)]
        if Feature.UNICODE_CATEGORY in self.features:
            features.append(self.unicode_features(paragraph))
        if Feature.LETTERS_COUNT in self.features:
            features.append(self.letters_count(paragraph))

        return torch.concat(features)

    def process_sequence(self, sequence: List[str]):
        features = []
        for sentence in tqdm(sequence):
            features.append(self.process_sentence(sentence))
        return torch.stack(features)

    def __call__(self, data: Union[str, List[str]]):
        if isinstance(data, str):
            return self.process_sentence(data)
        return self.process_sequence(data)

    def unicode_features(self, paragraph):
        data = torch.zeros(len(self.unicode_categories))

        cat_count = Counter(unicodedata.category(char) for char in paragraph)
        for idx, feature in enumerate(self.unicode_categories):
            count = cat_count.get(feature, 0)
            data[idx] = count

        # normalize the data to percentage of the sentence exists of
        if self.normalize:
            return data / torch.sum(data)
        else:
            return data

    def letters_count(self, paragraph):
        data = torch.zeros(len(self.letters))

        counter = Counter(char for char in paragraph)
        for idx, letter in enumerate(self.letters):
            data[idx] = counter.get(letter, 0)

        # normalize the data to percentage of the sentence exists of
        if torch.sum(data) == 0:
            return data
        return data / torch.sum(data)
