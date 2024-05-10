"""
@author: sijie
Created date: 06/05/2024
Update date: 06/05/2024
Causal LM Training Script
"""

from numpy import block
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelforCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModelling
from itertools import batched, chain
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

## Chunk tokenized_dataset using the chunk_text function
dataset = tokenized_dataset.map(chunk_text, batched=True)

## Setup DataCollator to be used for training
data_collator = DataCollatorForLanguageModelling(tokenizer=tokenizer,
                                                 mlm=False,
                                                 return_tensors="pt")
## mlm stands for masked language modelling. Set to True only if that's the task you want to learn
## return_tensors specify which type of tensor you want to be returned. pt == pytorch tensor

## Define training arguments
training_args = TrainingArguments(output_dir="finetune_gpt2")

## Define Trainer
trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=dataset,
                  data_collator=data_collator)

## Start training
trainer.train()

######################### Inference #########################################

## Specify input_string
input_string = "Cross-Site Scripting is a vulnerability that"

## tokenize input string
input_ids = tokenizer(input_string, return_tensors="pt").input_ids

## Inference: Generate model output_ids
outputs = model.generate(input_ids, num_beams=10,
                         num_return_sequences=1,
                         no_repeat_ngram_size=1,
                         remove_invalid_values=True)
## num_beams controls the number of beams (i.e., possible sequences) to consider in the beam search algorithm.
## num_return_sequences=1: This parameter controls the number of sequences to return.
## no_repeat_ngram_size: prevents the model from generating repeated sequences of tokens.
## remove_invalid_values: removes any invalid values from the output. The exact meaning of "invalid values" depends on the model, but it likely refers to tokens that are not in the model's vocabulary or are otherwise invalid.

## Decode the output tokens into text
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(output_text)
