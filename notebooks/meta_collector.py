# Try to do this in parallel to the tokenizer
# Make a code that takes a list of sentences and converts them into metadata of choice
import unicodedata

import torch


def metadata_collector(
    sentences,
    device,
    features=["Ll", "Zs", "Lu", "Po", "Pd", "Lo", "Mn", "Ps", "Pe", "Mc"],
):
    data = torch.zeros((len(sentences), len(features))).to(device)

    for i, paragraph in enumerate(sentences):
        for char in paragraph:
            cat = unicodedata.category(char)

            #  if cat in ["Ll", "Zs", "Lu", "Po", "Pd", "Lo", "Mn", "Ps", "Pe", "Mc"]
            for idx in range(len(features)):
                if cat in features[idx]:
                    data[i][idx] += 1

    # normalize the data to percentage of the sentence exists of
    return torch.div(data.T, torch.sum(data, 1)).T


#     return data
