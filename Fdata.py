import os
import shutil

# 原始数据集路径
src_root = r'C:\Users\admin\Desktop\Flickr1024 Dataset'
# 目标路径
dst_root = r'C:\Users\admin\Desktop\Flickr1024'

# 处理所有三个数据集分割
for split in ['test', 'train', 'val']:
    src_dir = os.path.join(src_root, split)
    dst_dir = os.path.join(dst_root, split, 'gt')

    # 遍历源目录中的所有文件
    for filename in os.listdir(src_dir):
        if filename.endswith(('_L.png', '_R.png')):
            # 解析文件名和左右标识
            base_name = filename.split('_')[0]
            suffix = filename.split('_')[-1]
            lr = suffix[0]  # 取L或R

            # 创建目标路径
            target_subdir = os.path.join(dst_dir, 'left' if lr == 'L' else 'right')
            os.makedirs(target_subdir, exist_ok=True)

            # 生成新文件名（替换L/R为001）
            new_filename = f"{base_name}_001.png"

            # 完整路径
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(target_subdir, new_filename)

            # 执行复制操作
            shutil.copy2(src_path, dst_path)
            print(f'Copied: {src_path} -> {dst_path}')

print("所有文件已处理完成！")