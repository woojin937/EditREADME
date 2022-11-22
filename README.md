## BERT implementation process

My project contains 1) tokenization.ipynb and 2) BERT.ipynb file.

## Load dataset - Tokenization.ipynb

To download the wikitext-103-raw-v1 dataset I used HuggingFace’s datasets library (https://huggingface.co/datasets/wikitext) — you can install the dataset with pip install datasets. Then we download wikitext-103-raw-v1 dataset with:

```python
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
```
This code create text.vocab, text.txt, text.model, text.json. Set variable "data_dir" to the path to the directory where you want to store the generated file.

## Load dataset - BERT.ipynb

This code create text_bert_0.json file. Set variable "data_dir" to the path to the directory where you want to store the generated file.

## Reference code

1) tokenization.ipynb file - Get vocab generation code from https://github.com/paul-hyun/transformer-evolution/blob/master/tutorial/vocab_with_sentencepiece.ipynb.

2) BERT.ipynb file - Use my original code for pretrain data generation. Get other codes (encoding, BERT, BERT pretrain, pretrain dataset) from https://github.com/paul-hyun/transformer-evolution/blob/master/tutorial/bert-01.ipynb.


