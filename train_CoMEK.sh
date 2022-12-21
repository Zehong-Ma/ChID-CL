export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 --nnodes=1 --start_method=spawn idom_token_together.py \
--model_name_or_path hfl/chinese-roberta-wwm-ext \
--do_train \
--do_eval \
--train_file ./data/train_data.json \
--validation_file ./data/dev_data.json \
--test_file ./data/test_data.json \
--idiom_file ./data/idiom_processed.json \
--metric_for_best_model eval_accuracy \
--load_best_model_at_end \
--learning_rate 5e-5 \
--evaluation_strategy epoch \
--num_train_epochs 5 \
--logging_steps 50 \
--output_dir ./tmp \
--per_device_eval_batch_size 7 \
--per_device_train_batch_size 7 \
--seed 42 \
--max_seq_length 512 \
--warmup_ratio 0.1 \
--save_strategy epoch \
--overwrite_output 

# torchrun --nproc_per_node=8 --nnodes=1 --start_method=spawn idom_token_together.py \
# --model_name_or_path hfl/chinese-roberta-wwm-ext \
# --do_train \
# --do_eval \
# --train_file ./data/train_data_10w.json \
# --validation_file ./data/dev_data.json \
# --test_file ./data/test_data.json \
# --idiom_file ./data/idiom_processed.json \
# --metric_for_best_model eval_accuracy \
# --load_best_model_at_end \
# --learning_rate 5e-5 \
# --evaluation_strategy epoch \
# --num_train_epochs 5 \
# --logging_steps 50 \
# --output_dir ./tmp \
# --per_device_eval_batch_size 7 \
# --per_device_train_batch_size 7 \
# --seed 42 \
# --max_seq_length 512 \
# --warmup_ratio 0.1 \
# --save_strategy epoch \
# --overwrite_output 

torchrun --nproc_per_node=8 --nnodes=1 --start_method=spawn idom_token_together.py \
--model_name_or_path hfl/chinese-roberta-wwm-ext \
--do_train \
--do_eval \
--train_file ./data/train_data_5w.json \
--validation_file ./data/dev_data.json \
--test_file ./data/test_data.json \
--idiom_file ./data/idiom_processed.json \
--metric_for_best_model eval_accuracy \
--load_best_model_at_end \
--learning_rate 5e-5 \
--evaluation_strategy epoch \
--num_train_epochs 5 \
--logging_steps 50 \
--output_dir ./tmp \
--per_device_eval_batch_size 7 \
--per_device_train_batch_size 7 \
--seed 42 \
--max_seq_length 512 \
--warmup_ratio 0.1 \
--save_strategy epoch \
--overwrite_output

# 1w

torchrun --nproc_per_node=8 --nnodes=1 --start_method=spawn idom_token_together.py \
--model_name_or_path hfl/chinese-roberta-wwm-ext \
--do_train \
--do_eval \
--train_file ./data/train_data_1w.json \
--validation_file ./data/dev_data.json \
--test_file ./data/test_data.json \
--idiom_file ./data/idiom_processed.json \
--metric_for_best_model eval_accuracy \
--load_best_model_at_end \
--learning_rate 5e-5 \
--evaluation_strategy epoch \
--num_train_epochs 5 \
--logging_steps 50 \
--output_dir ./tmp \
--per_device_eval_batch_size 7 \
--per_device_train_batch_size 7 \
--seed 42 \
--max_seq_length 512 \
--warmup_ratio 0.1 \
--save_strategy epoch \
--overwrite_output