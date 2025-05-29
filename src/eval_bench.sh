#!/bin/bash
# run_models.sh

model_paths=(
    "Model Path"
)

file_names=(
    "File Name"
)

export DECORD_EOF_RETRY_MAX=40960 

for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    file_name="${file_names[$i]}"
    nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python ./src/eval_bench1.py --model_path "$model" --file_name "$file_name" > eval_${file_name}.log 2>&1 &
done
