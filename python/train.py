import os
import json
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig
from create_data import SquadExample, create_squad_examples, create_inputs_targets
from models import create_model
from utils import ExactMatch


# code based on:
# https://keras.io/examples/nlp/text_extraction_with_bert/

max_len = 384
configuration = BertConfig()  # default parameters and configuration for BERT

## set-up BERT tokenizer
# Save the slow pretrained tokenizer
slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
save_path = "bert_base_uncased/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)

## load the data
train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
train_path = keras.utils.get_file("train.json", train_data_url)
eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
eval_path = keras.utils.get_file("eval.json", eval_data_url)

## preprocess the data
with open(train_path) as f:
    raw_train_data = json.load(f)

with open(eval_path) as f:
    raw_eval_data = json.load(f)

train_squad_examples = create_squad_examples(tokenizer, raw_train_data, max_len)
x_train, y_train = create_inputs_targets(train_squad_examples)
print(f"{len(train_squad_examples)} training points created.")

eval_squad_examples = create_squad_examples(tokenizer, raw_eval_data, max_len)
x_eval, y_eval = create_inputs_targets(eval_squad_examples)
print(f"{len(eval_squad_examples)} evaluation points created.")

## create model
use_tpu = False  # default: True (will use TPU)
if use_tpu:
    # Create distribution strategy
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # Create model
    with strategy.scope():
        model = create_model(max_len)
else:
    model = create_model(max_len)

model.summary()

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(lr=5e-5)
model.compile(optimizer=optimizer, loss=[loss, loss])

## train and evaluate
exact_match_callback = ExactMatch(x_eval, y_eval)
model.fit(
    x_train,
    y_train,
    epochs=1,  # For demonstration, 3 epochs are recommended
    verbose=2,
    batch_size=64,  # 64
    callbacks=[exact_match_callback],
)


