#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

from accelerate import Accelerator, DistributedType

import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict, disable_caching, load_from_disk
from spacy.lang.en import English 

nlp = English()
nlp.add_pipe('sentencizer')

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


# TODO: chage comments
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_output_dir:str = field(
        metadata={
            "help": "Directory to save the preprocessed dataset."
        }
    )
    validation_split_percentage: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    do_split_in_sentences:bool = field(
        default=True,
        metadata = {
            "help": "Use or not spacy sentencizer to split in sentences."
        }
    )
    few_special_tokens:bool = field(
        default=False,
        metadata={
            "help": "If true only add [SEP] at the end of sequence."
        }
    )


def split_in_sentences(text_list):
    docs = [nlp(text) for text in text_list]
    return [str(sent).strip() for doc in docs for sent in doc.sents if len(str(sent)) > 10]


def main():
    # disable caching since this script will be run once time 
    # disable_caching()

    # import datasets
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()


    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator()

    max_seq_length = data_args.max_seq_length
    few_special_tokens = data_args.few_special_tokens

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # CUSTOM DATASET SETTING

    joint_datasets = [
        load_dataset(
            "wikipedia", 
            "20220301.en",
            cache_dir=model_args.cache_dir
        )["train"].remove_columns(["id", "url", "title"]),
        load_dataset("bookcorpus", cache_dir=model_args.cache_dir)["train"]
    ]

    raw_datasets = concatenate_datasets(joint_datasets).train_test_split(test_size=data_args.validation_split_percentage)
    raw_datasets["validation"] = raw_datasets["test"]
    del raw_datasets["test"]

    for i in raw_datasets:
        raw_datasets[i] = raw_datasets[i].shuffle(42)

    do_split_in_sentences = data_args.do_split_in_sentences

    if do_split_in_sentences:
        with accelerator.main_process_first():
            splat_raw_datasets = raw_datasets.map(
                lambda batch: {"text":split_in_sentences(batch["text"])},
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                desc="Splitting on sentences.",
            )
    else:
        splat_raw_datasets = raw_datasets

    # --- TOKENIZER
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", **tokenizer_kwargs)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = splat_raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.
    def tokenize_function(examples):
        tokenized = {k:i for k,i in tokenizer(examples[text_column_name], return_special_tokens_mask=True).items()}
        return tokenized

    with accelerator.main_process_first():
        tokenized_datasets = splat_raw_datasets.map(
            lambda x: tokenize_function(x),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on every text in dataset"
        )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def _join_separate_sentences(concatenated_examples, pad):
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        local_max_seq_length = max_seq_length - pad
        if total_length >= local_max_seq_length:
            total_length = (total_length // local_max_seq_length) * local_max_seq_length
        # Split by chunks of max_len.
        return {
            k: [t[i : i + local_max_seq_length] for i in range(0, total_length, local_max_seq_length)]
            for k, t in concatenated_examples.items()
        }

    def group_texts_plus_cls(examples):
        # Concatenate all texts. After removing first token ([CLS]/<s>).
        concatenated_examples = {k: list(chain(*[i[1:] for i in examples[k]])) for k in examples.keys()}
        result = _join_separate_sentences(concatenated_examples, 1)
        # Add back [CLS]/<s> only at the beginning of the first sentence (e.g. only at the beginnning
        # of the whole sequence)
        return {
            "input_ids":[[tokenizer.cls_token_id] + i for i in result["input_ids"]],
            "attention_mask":[[1] + i for i in result["attention_mask"]],
            "token_type_ids":[[0] + i for i in result["token_type_ids"]],
            "special_tokens_mask":[[1] + i for i in result["special_tokens_mask"]]
        }

    def group_texts_plus_cls_and_sep(examples):
        # Concatenate all texts. After removing first token ([CLS]/<s>).
        concatenated_examples = {k: list(chain(*[i[1:-1] for i in examples[k]])) for k in examples.keys()}
        result = _join_separate_sentences(concatenated_examples, 2)
        # Add back [CLS]/<s> only at the beginning of the first sentence (e.g. only at the beginnning
        # of the whole sequence) and [SEP]/</s> only at the end of the whole sequence.
        return {
            "input_ids":[[tokenizer.cls_token_id] + i + [tokenizer.sep_token_id] for i in result["input_ids"]],
            "attention_mask":[[1] + i + [1] for i in result["attention_mask"]],
            "token_type_ids":[[0] + i + [0] for i in result["token_type_ids"]],
            "special_tokens_mask":[[1] + i + [1] for i in result["special_tokens_mask"]]
        }

    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    with accelerator.main_process_first():
        joint_tokenized_datasets = tokenized_datasets.map(
            group_texts_plus_cls if not few_special_tokens else group_texts_plus_cls_and_sep,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )

    # write to disk the preprocessed data
    print(f"Preprocessing finished, saving dataset into disk {data_args.dataset_output_dir}")
    joint_tokenized_datasets.save_to_disk(data_args.dataset_output_dir)


if __name__ == "__main__":
    main()
