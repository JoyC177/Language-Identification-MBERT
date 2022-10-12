# [UvA MSc AI DL4NLP] Project: Written Language Identification using MBERT and Engineered Language Features

This code supports our report of the same name.

## Authors

- Konrad Bereda
- Joy Crosbie
- Nils Peters
- Noah van der Vleuten

## Setup

Configure your Conda environment using the `environment.yml` file.

## Usage

### Finetuning the models
To finetune the models we have created a seperate notebook in the following path: `notebooks/finetuning_(xl)mbert.py`.

All models in the report that have been finetuned have been trained in this notebook. 
This was done such that the main training framework in this repository would not have to be overhauled specifically for finetuning.
The notebook provides explanation on how to run it.