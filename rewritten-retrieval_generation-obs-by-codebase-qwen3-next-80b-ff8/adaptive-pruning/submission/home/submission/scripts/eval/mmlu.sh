#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
# zero-shot
model_name_or_path=$1

mkdir -p output/results/mmlu/llama-7B-5shot/

python run_eval_llama_mmlu.py \
    --ntrain 5 \
    --data_dir /mmfs1/home/bowen98/projects/AdaptPruning/data/eval/mmlu \
    --output_dir output/results/mmlu/llama2-7B-0shot/ \
    --model_name_or_path ${model_name_or_path} \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --eval_batch_size 2 | tee "${model_name_or_path}/mmlu-5shot.log"