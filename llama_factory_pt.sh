
export USE_MODELSCOPE_HUB=1
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
--stage pt \
--do_train \
--model_name_or_path  AI-ModelScope/bloomz-560m \
--dataset wiki_demo \
--finetuning_type lora \
--lora_target query_key_value \
--output_dir path_to_pt_checkpoint \
--overwrite_cache \
--per_device_train_batch_size 3 \
--gradient_accumulation_steps 4 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 1000 \
--learning_rate 5e-5 \
--num_train_epochs 3.0 \
--plot_loss \
--fp16 \
--overwrite_output_dir
