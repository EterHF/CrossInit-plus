# export LD_PRELOAD="/home/node-user/anaconda3/envs/ci/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12"
export CUDA_VISIBLE_DEVICES=0

python test_cross_init.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" \
    --num_inference_steps 50 \
    --learned_embedding_path "./results/9/learned_embeds.bin" \
    --prompt "a {} person on the beach" \
    --save_dir "./tmp" \
    --num_images_per_prompt=8 \
    --n_iter=1