import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from ..utils import DEVICE

MBERT_MODEL = "bert-base-multilingual-cased"
XLMBERT_MODEL = "xlm-roberta-base"
BERTIC_MODEL = "classla/bcms-bertic"
EMBEDDINGS_DIR = "embeddings/"
NUM_LAYERS = 12


def generate_bert_embeddings(
    model, tokenizer, dataset: Dataset, batch_size: int = 4
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
                list(sentences_batch),
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(DEVICE)
            output = model(**encoded_input)

            # Take for the 0 layer the average of the tokens minus the CLS one
            stacked = torch.stack(output.hidden_states)
            stacked[0, :, 0, :] = (torch.sum(stacked[0], 1) - stacked[0, :, 0, :]) / (
                stacked.shape[2] - 1
            )

            # Take [CLS] token embedding of specific layers or the last one.
            cls_embeddings = stacked[:, :, 0, :].cpu()

            # Store the embeddings
            embeddings.append(cls_embeddings)

    return torch.concat(embeddings, dim=1)


def load_model(model_name: str = MBERT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(DEVICE)

    return model, tokenizer
