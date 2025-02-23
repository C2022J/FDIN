from PIL import Image

# 需要裁剪的图片路径列表及目标尺寸
file_paths = [
    (r"C:\Users\admin\Desktop\Flickr1024\train\gt\right\199_001.png", (502, 1000)),  # 503x1000 -> 502x1000
    (r"C:\Users\admin\Desktop\Flickr1024\train\input\right\199_001.png", (251, 500)),  # 252x500 -> 251x500
    (r"C:\Users\admin\Desktop\Flickr1024\train\gt\right\504_001.png", (600, 1102)),  # 600x1103 -> 600x1102
    (r"C:\Users\admin\Desktop\Flickr1024\train\input\right\504_001.png", (300, 551)),  # 300x552 -> 300x551
    (r"C:\Users\admin\Desktop\Flickr1024\train\input\left\237_001.png", (450, 550)),  # 450x551 -> 450x550
    (r"C:\Users\admin\Desktop\Flickr1024\train\gt\left\237_001.png", (900, 1100)),  # 900x1101 -> 900x1100
]

def crop_to_target_size(image_path, target_size):
    """
    将图片裁剪为目标尺寸
    :param image_path: 图片路径
    :param target_size: 目标尺寸 (width, height)
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            target_width, target_height = target_size

            if (width, height) != (target_width, target_height):
                # 裁剪图片
                img_cropped = img.crop((0, 0, target_width, target_height))
                img_cropped.save(image_path)  # 覆盖原图
                print(f"裁剪完成: {image_path} (新尺寸: {target_width}x{target_height})")
            else:
                print(f"无需裁剪: {image_path} (尺寸已为目标尺寸)")
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")

if __name__ == "__main__":
    for file_path, target_size in file_paths:
        crop_to_target_size(file_path, target_size)