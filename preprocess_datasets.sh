#!/bin/bash

python preprocess_datasets.py --dataset_output_dir wiki_book_sentences_dataset \
     --validation_split_percentage 0.1 \
     --preprocessing_num_workers 8 \
     --max_seq_length 256 \
     --do_split_in_sentences true \
     --cache_dir hf_cache