import os
import shutil

# 图片文件夹路径
image_dir = "/home/node-user/traffic/yemao/CrossInit/CelebAMask-HQ/CelebA-HQ-img"  # 替换为实际路径

# 获取文件夹内所有jpg文件
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# 遍历所有的图片文件
for image_file in image_files:
    # 提取文件名中的数字部分作为文件夹名
    folder_name = os.path.splitext(image_file)[0]  # 去掉'.jpg'后缀

    # 创建新文件夹
    folder_path = os.path.join(image_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # 构造图片的完整路径
    image_path = os.path.join(image_dir, image_file)

    # 将图片移动到新文件夹中
    shutil.move(image_path, os.path.join(folder_path, image_file))

    print(f"Moved {image_file} to {folder_path}")

print("All images have been moved to their respective folders.")