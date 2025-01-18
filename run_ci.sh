# export LD_PRELOAD="/home/node-user/anaconda3/envs/ci/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12"
export CUDA_VISIBLE_DEVICES=$3

base_dir="/home/node-user/traffic/yemao/CrossInit/CelebAMask-HQ/CelebA-HQ-img"  # 替换为实际路径

lb=$1
ub=$2

# 使用 find 查找所有子文件夹并按字母顺序排序
folders=($(find "$base_dir" -mindepth 1 -maxdepth 1 -type d | sort))

# 遍历每个文件夹
for (( i=lb; i<ub; i++ )); do
    folder="$base_dir/$i"  # 获取文件夹
    # 确保只处理文件夹
    if [ -d "$folder" ]; then
    #     # 获取文件夹的名称（即ID）
    #     folder_name=$(basename "$folder")

        # 设置 train_data_dir 和 output_dir
        train_data_dir="$folder"
        output_dir="./results/$i"
        
        # 创建 output_dir 文件夹（如果不存在）
        mkdir -p "$output_dir"

        # 将 train_data_dir 中的图片复制到 output_dir
        echo "Copying images from $train_data_dir to $output_dir"
        cp "$train_data_dir"/*.jpg "$output_dir"  # 只复制jpg文件，如果需要复制其他格式，可以修改

        # 执行命令
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