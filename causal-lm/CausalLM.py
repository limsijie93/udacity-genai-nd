"""
@author: sijie
Created date: 06/05/2024
Update date: 06/05/2024
Causal LM Training Script
"""

from numpy import block
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelforCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModelling
from itertools import chain
import pandas as pd
from pathlib import Path

def tokenize_function(examples):
    """
    Wrapper function to tokenize input examples for
        - input_ids
        - attention_mask
        - labels (if available)
    """
    output = tokenizer(examples["text"])
    return output

DATA_PATH = Path("./data")
file_paths = [filename for filename in data_path.glob("*.txt")]

## Read all the files into a single list
file_data = list()
for filename in file_paths:
    with open(filename, "r") as f:
        data = f.read()
    file_data.append(data)

## Convert list of text into Dataset
dataset = Dataset.from_dict({"text": file_data})

## Define MODEL_NAME
MODEL_NAME = "gpt2"

## Load Tokenizer
tokenizer = AutoTokenizer.from_pretrainer(MODEL_NAME)
### add pad token since GPT2 doesn't have a pad token
tokenizer.pad_token = tokenizer.eos_token

## Load Model
model = AutoModelforCausalLM.from_pretrained(MODEL_NAME)

## Tokenize dataset using map function
tokenized_dataset = dataset.map(tokenize_function, batched=True)
## Remove the original data column that has already been tokenized
tokenized_dataset = tokenized_dataset.remove_columns(dataset.column_names)


def chunk_text(examples, block_size=1024):
    """
    Chunks text examples into fixed-size blocks.

    Args:
        examples (dict): A dictionary containing lists of text examples.
            Each key represents a category, and each value is a list of text examples.
        block_size (int, optional): The size of the chunks to split the text into. Defaults to 1024.

    Returns:
        dict: A dictionary containing the chunked text examples.
            Each key represents a category, and each value is a list of text chunks.
            The "labels" key is a copy of the "input_ids" key.

    Notes:
        This function concatenates all text examples for each category, computes the total length,
        and then splits the text into chunks of the specified block size.
    """

    ## concat all the texts together for each example
    concatenated_examples = dict()
    for k in examples.keys():
        concatenated_examples[k] = list(chain(*examples[k]))

    ## compute total_length of all the texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    ## drop the remainder of the block + return 0 if total_length < block_size
    total_length = (total_length // block_size) * block_size

    ## split into chunks of block_size
    result = dict()
    for k, t in concatenated_examples.items():
        ## divide each text into chunks of 1024
        chunks = list()
        for i in range(total_length, block_size):
            chunks.append(t[i: i + block_size])
        result[k] = chunks
    result["labels"] = result["input_ids"].copy()
    return result
