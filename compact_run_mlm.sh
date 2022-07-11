#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_HiNLM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --partition=m100_usr_prod
#SBATCH --job-name=pretraining_BERT
#SBATCH --error=pretraining_BERT.error
#SBATCH --output=pretraining_BERT.out
#SBATCH --constraint=gpureport
#SBATCH --mem=20000

python -m torch.distributed.launch --nproc_per_node 4 compact_run_mlm.py \
--output_dir mlm_run_BERT \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 8 \
--warmup_step 30000 \
--weight_decay 0.01 \
--num_train_epochs 3 \
--learning_rate 1.e-4 \
--config_name model_config/my_bert_conf.json \
--dataset_folder wiki_book_sentences_dataset \
--cache_dir hf_cache \
--seed 2147483647 \
--save_steps 10000 \
--do_eval true \
--ddp_find_unused_parameters False \

# commentare nella prima run
# --resume_from_checkpoint /m100_work/IscrC_HiNLM/idini000/modules/transformers/examples/pytorch/language-modeling/pretraining-checkpoints-all/checkpoint-580000