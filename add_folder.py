import os
import shutil
import argparse

def organize_images(image_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    for image_file in image_files:
        folder_name = os.path.splitext(image_file)[0]  # 去掉 '.jpg' 后缀
        folder_path = os.path.join(image_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)  # 创建文件夹（若不存在）

        image_path = os.path.join(image_dir, image_file)
        shutil.move(image_path, os.path.join(folder_path, image_file))  # 移动文件

        print(f"Moved {image_file} to {folder_path}")

    print("All images have been moved to their respective folders.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize images into folders based on their filenames.")
    parser.add_argument("image_dir", type=str, help="Path to the directory containing the images.", required=True)

    args = parser.parse_args()

    organize_images(args.image_dir)