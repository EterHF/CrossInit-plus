# export LD_PRELOAD="/home/node-user/anaconda3/envs/ci/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12"
export CUDA_VISIBLE_DEVICES=0
python train_cross_init.py \
    --save_steps 50000 \
    --checkpointing_steps 50000 \
    --only_save_embeds \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base"\
    --train_data_dir "./examples/input_images/28068" \
    --placeholder_token "<189469>" \
    --initialize_tokens "human face" \
    --n_persudo_tokens 2 \
    --reg_weight 0 \
    --repeats 100 \
    --output_dir "./logs/28068" \
    --logging_dir "logs" \
    --train_batch_size 8 \
    --max_train_steps 2000 \
    --learning_rate 0.000625 \
    --scale_lr \
    --validation_prompt "a {} person and Anne Hathaway enjoying a day at an amusement park" \
    --num_validation_images 4 \
    --validation_steps 50000 \