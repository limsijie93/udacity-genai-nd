#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install evaluate
# !pip install scikit-learn
# !pip install trl


# # Lightweight Fine-Tuning Project

# TODO: In this cell, describe your choices for each of the following
# 
# * PEFT technique: 
# * Model: 
# * Evaluation approach: 
# * Fine-tuning dataset: 

# In[2]:


## PEFT examples: https://github.com/huggingface/peft/tree/main/examples


# ## Loading and Evaluating a Foundation Model
# 
# TODO: In the cells below, load your chosen pre-trained Hugging Face model and evaluate its performance prior to fine-tuning. This step includes loading an appropriate tokenizer and dataset.

# In[3]:


import torch
import numpy as np

import evaluate
from evaluate import evaluator
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType, AutoPeftModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split


# In[4]:


MODEL_NAME = "distilbert-base-uncased"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128


# In[5]:


dataset = load_dataset("sms_spam", split="train")
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)


# In[6]:


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# In[7]:


def tokenize(examples):
    # return_tensors="pt" ensures that the tokenized output is in pytorch tensors
    # truncation=True ensures that all input into the model has consistent size. 
    # Padded/truncated to the max_length of the model
    
    return tokenizer(examples["sms"], padding="max_length", 
                     truncation=True, return_tensors="pt")

def compute_metrics(eval_pred):
    """
    Wrapper method to do calculation for metrics that we are interested in
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    eval_loss = torch.nn.CrossEntropyLoss()(torch.tensor(logits), torch.tensor(labels))
    metrics = metric.compute(predictions=predictions, references=labels)
    metrics['eval_loss'] = eval_loss.item()
    return metrics


# In[8]:


metric = evaluate.load("accuracy", "cross_entropy")


# In[9]:


tokenized_dataset = dataset.map(tokenize, batched=True)


# In[10]:


model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2,
                                                          id2label={0: "not spam", 1: "spam"},
                                                          label2id={"not spam": 0, "spam": 1})


# In[11]:


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)


# In[12]:


# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# In[ ]:


trainer.evaluate()


# ## Performing Parameter-Efficient Fine-Tuning
# 
# TODO: In the cells below, create a PEFT model from your loaded model, run a training loop, and save the PEFT model weights.

# In[ ]:


## Specifying the task_type will then create the relevant instance of the object
## Reference: https://github.com/huggingface/peft/blob/02ae6bcb373d9d9d3bec9ba920d63316418ff64a/src/peft/peft_model.py#L1094C7-L1094C41

## Available task types: https://huggingface.co/docs/peft/en/package_reference/peft_types
lora_config = LoraConfig(task_type="SEQ_CLS", r=4, lora_alpha=1, 
                         lora_dropout=0, target_modules=["pre_classifier", "classifier"],
                         inference_mode=False)


# In[ ]:


peft_lora_model = get_peft_model(model, lora_config)
peft_lora_model.print_trainable_parameters()


# In[ ]:


peft_lora_model


# In[ ]:


# Define training arguments
peft_training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=15,
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=16,
    evaluation_strategy='epoch',
    logging_strategy='epoch',
#     eval_steps=5,
#     logging_steps=5,
    save_strategy='epoch',
    save_safetensors=False,
    ## https://github.com/huggingface/transformers/issues/27613#issuecomment-1848645557
    
    warmup_steps=500,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    
    
    # Can play with remove_unused_columns. Initially set this to False because Trainer is returning 
    # IndexError: Invalid key: 4437 is out of bounds for size 0
    # https://discuss.huggingface.co/t/indexerror-invalid-key-16-is-out-of-bounds-for-size-0/14298/4
    remove_unused_columns=False,
#     label_names=['label']
    
    # By default, Trainer uses GPU on the device
    # However, if you want to explicitly set GPU device(s)
    # no_cuda=False,  # Set to False to enable GPU usage
    # device=[0],  # Use GPU device with index 0, in case you have multiple GPUs
)


# In[ ]:


# Create a Trainer instance
peft_trainer = Trainer(
    model=peft_lora_model,
    args=peft_training_args,

    # We are dropping the SMS column because the size of this input column is not consistent
    # Not removing this column will lead to 
    # ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (`sms` in this case) have excessive nesting (inputs type `list` where type `int` is expected).
    train_dataset=tokenized_dataset["train"].remove_columns(["sms"]),
    eval_dataset=tokenized_dataset["test"].remove_columns(["sms"]),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
#.rename_column("label", "labels")


# In[ ]:


## https://stackoverflow.com/questions/76012700/validation-loss-shows-no-log-during-fine-tuning-model


# In[ ]:


# Train the model
peft_trainer.train()


# In[ ]:


# Save the trained model
peft_trainer.save_model(f"./results/{MODEL_NAME}-best")


# ## Performing Inference with a PEFT Model
# 
# TODO: In the cells below, load the saved PEFT model weights and evaluate the performance of the trained PEFT model. Be sure to compare the results to the results from prior to fine-tuning.

# In[ ]:


output_dir = f"./results/{MODEL_NAME}-best"


# In[ ]:


saved_tokenizer = AutoTokenizer.from_pretrained(output_dir)
saved_model = AutoPeftModelForSequenceClassification.from_pretrained(output_dir)


# In[ ]:


# Create a Trainer instance
saved_trainer = Trainer(
    model=saved_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# In[ ]:


saved_trainer.evaluate()


# In[ ]:




