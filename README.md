# [UvA MSc AI DL4NLP] Project: Written Language Identification using MBERT and Engineered Language Features

This code supports our report of the same name. It can be used to train and test Transformer-based language models on the task of Language Identification, with additional metadata features.

## Authors

- Konrad Bereda
- Joy Crosbie
- Nils Peters
- Noah van der Vleuten

## Setup

Configure your Conda environment using the `environment.yml` file.
```bash
conda env create --file environment.yml
```

## Usage
The main functionality is accessible through `train_eval.py` script. To see all options available, you can use following command:
```bash
python train_eval.py --help
```
Below you can find an example command that trains M-BERT model with Unicode features:
```bash
WANDB_MODE=offline python train_eval.py \
                     --features UNICODE_CATEGORY \
                     --normalize true \
                     --bert-model-name bert-base-multilingual-cased \
                     --batch-size 1024 \
                     --optimizer SGD \
                     --learning-rate 0.2
```
We use [wandb.ai](https://wandb.ai) to track our experiments. If you don't have it configured, you can temporarily disable it using environmental variable `WANDB_MODE=offline`.
### Finetuning the models
To finetune the models we have created a separate notebook in the following path: `notebooks/finetuning_(xl)mbert.py`.

All models in the report that have been finetuned have been trained in this notebook. 
This was done such that the main training framework in this repository would not have to be overhauled specifically for finetuning.
The notebook provides explanation on how to run it.