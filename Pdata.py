import os

def rename_files(directory):
    # 遍历指定目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否以 '_000.png' 结尾
        if filename.endswith('_000.png'):
            # 构建新的文件名
            new_filename = filename.replace('_000.png', '_001.png')
            # 获取文件的完整路径
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # 重命名文件
            os.rename(old_file, new_file)
            print(f'Renamed: {filename} -> {new_filename}')

# 指定目录路径
directory_path = r'C:\Users\admin\Desktop\Flickr1024_Middlebury\train\input\left'

# 调用函数进行重命名
rename_files(directory_path)