from typing import Tuple

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer

from ..utils import DEVICE

MBERT_MODEL = "bert-base-multilingual-cased"
EMBEDDINGS_DIR = "embeddings/"
EMBEDDINGS_FNAME = "bert_embeddings_{}.npy"


def generate_bert_embeddings(
    model: BertModel, tokenizer: BertTokenizer, dataset: Dataset, batch_size: int = 4
) -> torch.Tensor:
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    with torch.no_grad():
        embeddings = []
        # Loop over the sentences in batches
        for sentences_batch, _ in tqdm(data_loader):
            encoded_input = tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(DEVICE)
            output = model(**encoded_input)
            # Take [CLS] token embedding
            last_hidden_states = output[0][:, 0, :].cpu()
            # Store the embeddings
            embeddings.append(last_hidden_states)
    return torch.concat(embeddings)


def load_model(model_name: str = MBERT_MODEL) -> Tuple[BertModel, BertTokenizer]:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(DEVICE)

    return model, tokenizer
