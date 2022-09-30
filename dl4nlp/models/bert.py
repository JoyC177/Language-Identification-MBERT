from typing import Tuple

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer
from transformers import XLMRobertaTokenizer, XLMRobertaModel

from ..utils import DEVICE

MBERT_MODEL = "bert-base-multilingual-cased"
XLMBERT_MODEL = "xlm-roberta-base"
EMBEDDINGS_DIR = "embeddings/"
NUM_LAYERS = 12


def generate_bert_embeddings(
    model: BertModel, tokenizer: BertTokenizer, dataset: Dataset, batch_size: int = 4
) -> torch.Tensor:
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    layer_flag = None
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
            #print(output)
            # Take [CLS] token embedding of specific layers or the last one.
            if output.hidden_states:
                cls_embeddings = torch.stack(output.hidden_states)[:, :, 0, :].cpu()
                layer_flag = True
            else:
                cls_embeddings = output.last_hidden_state[:, 0, :].cpu()
            
            # Store the embeddings
            embeddings.append(cls_embeddings)

    if layer_flag:
        return torch.concat(embeddings, dim=1)
    else:
        return torch.concat(embeddings)


def load_model(model_name: str = MBERT_MODEL) -> Tuple[BertModel, BertTokenizer]:
    if model_name == MBERT_MODEL:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name).to(DEVICE)
    elif model_name == XLMBERT_MODEL:
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        model = XLMRobertaModel.from_pretrained(model_name).to(DEVICE)
    else:
        raise ValueError("Invalid bert-model-name was provided,  \
            try with 'xlm-roberta-base' or 'bert-base-multilingual-cased'")

    return model, tokenizer
