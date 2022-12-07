export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8

torchrun --nproc_per_node=8 --nnodes=1 --start_method=spawn ./embedding/embedding_pretrain.py \
--model_name_or_path hfl/chinese-roberta-wwm-ext \
--do_eval \
--train_file ./data/idiom_processed.json \
--validation_file ./data/idiom_processed.json \
--test_file ./data/idiom_processed.json \
--metric_for_best_model eval_accuracy \
--load_best_model_at_end \
--learning_rate 0 \
--evaluation_strategy epoch \
--num_train_epochs 1 \
--logging_steps 50 \
--output_dir ./embedding/ \
--per_device_eval_batch_size 16 \
--per_device_train_batch_size 16 \
--seed 42 \
--max_seq_length 512 \
--warmup_ratio 0.1 \
--save_strategy epoch \
--overwrite_output 