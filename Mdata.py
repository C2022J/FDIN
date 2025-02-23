import os
import shutil

# 源文件夹路径
source_folder = r"D:\BaiduNetdiskDownload\iPASSR_trainset\Middlebury"
# 左图目标文件夹路径
left_target_folder = r"C:\Users\admin\Desktop\Middlebury\train\gt\left"
# 右图目标文件夹路径
right_target_folder = r"C:\Users\admin\Desktop\Middlebury\train\gt\right"

# 创建目标文件夹（如果不存在）
os.makedirs(left_target_folder, exist_ok=True)
os.makedirs(right_target_folder, exist_ok=True)

# 用于记录不同场景的编号，从801开始
scene_number = 801
scene_dict = {}

# 遍历源文件夹中的文件
for filename in os.listdir(source_folder):
    if filename.endswith(".png"):
        base_name = filename.rsplit('.', 1)[0]  # 去除文件后缀
        prefix = base_name.split('_')[0]  # 获取文件名前面的数字部分

        if prefix not in scene_dict:
            scene_dict[prefix] = scene_number
            scene_number += 1

        if base_name.endswith("_L"):
            new_filename = f"{scene_dict[prefix]:03d}_000.png"
            target_folder = left_target_folder
        elif base_name.endswith("_R"):
            new_filename = f"{scene_dict[prefix]:03d}_001.png"
            target_folder = right_target_folder
        else:
            continue

        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, new_filename)
        shutil.copy2(source_path, target_path)