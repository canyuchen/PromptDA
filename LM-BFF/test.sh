CUDA_VISIBLE_DEVICES=0 python run.py \
    --task_name SST-2 \
    --data_dir data/k-shot/SST-2/16-42 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --model_name_or_path roberta-large \
    --few_shot_type prompt-demo \
    --num_k 16 \
    --max_steps 1000 \
    --eval_steps 100 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 0 \
    --output_dir result/tmp \
    --seed 42 \
    --template "*cls**sent_0*_It_was*mask*.*sep+*" \
    --mapping "{'0':'terrible','1':'great'}" \
    --num_sample 16 \


