# export LD_PRELOAD="/home/node-user/anaconda3/envs/ci/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12"
export CUDA_VISIBLE_DEVICES=$3

base_dir="/home/node-user/traffic/yemao/CrossInit/CelebAMask-HQ/CelebA-HQ-img"  # 替换为实际路径

lb=$1
ub=$2

folders=($(find "$base_dir" -mindepth 1 -maxdepth 1 -type d | sort))

for (( i=lb; i<ub; i++ )); do
    folder="$base_dir/$i"  
    # 确保只处理文件夹
    if [ -d "$folder" ]; then

        train_data_dir="$folder"
        output_dir="./results/$i"
        
        # 创建 output_dir 文件夹（如果不存在）
        mkdir -p "$output_dir"

        # 将 train_data_dir 中的图片复制到 output_dir
        echo "Copying images from $train_data_dir to $output_dir"
        cp "$train_data_dir"/*.jpg "$output_dir"
        
        python train_cross_init.py \
            --save_steps 500 \
            --only_save_embeds \
            --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" \
            --train_data_dir "$train_data_dir" \
            --placeholder_token "<sks>" \
            --celeb_path "./examples/wiki_names_v2.txt" \
            --n_persudo_tokens 2 \
            --reg_weight "1e-5" \
            --repeats 100 \
            --output_dir "$output_dir" \
            --logging_dir "logs" \
            --train_batch_size 8 \
            --max_train_steps 320 \
            --learning_rate 0.000625 \
            --scale_lr \
            --validation_prompt_file "prompts.txt" \
            --num_validation_images 4 \
            --validation_steps 320

        echo "Training for $folder_name completed."
    fi
done