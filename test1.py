import os
from PIL import Image

def check_odd_dimensions(root_dir):
    """
    检查图片的长或宽是否为奇数，并输出路径
    :param root_dir: Flickr1024 根目录路径
    """
    # 遍历所有子目录
    for split_dir in ['train', 'test', 'val']:
        for sub_dir in ['gt', 'input']:
            for lr_dir in ['left', 'right']:
                dir_path = os.path.join(root_dir, split_dir, sub_dir, lr_dir)
                if not os.path.exists(dir_path):
                    continue  # 如果目录不存在则跳过

                # 遍历目录下的所有图片文件
                for filename in os.listdir(dir_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        file_path = os.path.join(dir_path, filename)
                        try:
                            with Image.open(file_path) as img:
                                width, height = img.size
                                if width % 2 != 0 or height % 2 != 0:
                                    print(f"奇数尺寸图片: {file_path} (尺寸: {width}x{height})")
                        except Exception as e:
                            print(f"无法读取图片 {file_path}: {e}")

if __name__ == "__main__":
    flickr1024_path = r"C:\Users\admin\Desktop\Flickr1024"  # 修改为你的 Flickr1024 路径
    check_odd_dimensions(flickr1024_path)