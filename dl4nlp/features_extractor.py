import unicodedata
from collections import Counter
from enum import Enum, auto
from typing import List, Optional, Union

import torch
from tqdm.auto import tqdm

UNICODE_CATEGORIES = ["Ll", "Zs", "Lu", "Po", "Pd", "Lo", "Mn", "Ps", "Pe", "Mc"]


class Feature(Enum):
    UNICODE_CATEGORY = auto()


class FeaturesExtractor:
    def __init__(
        self,
        features: Optional[List[Union[Feature, str]]] = None,
        unicode_categories: List[str] = UNICODE_CATEGORIES,
        normalize = True
    ):
        if features is None:
            features = []

        for i, feature in enumerate(features):
            if isinstance(feature, str):
                features[i] = Feature[feature]

        self.features = features
        self.unicode_categories = unicode_categories
        self.normalize = normalize

    def process_sentence(self, paragraph: str):
        features = [torch.empty(0)]
        if Feature.UNICODE_CATEGORY in self.features:
            features.append(self.unicode_features(paragraph))

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
