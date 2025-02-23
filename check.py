import os
from PIL import Image


def check_images_folder(folder_path):
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    width, height = img.size
                    if width < 256 or height < 256:
                        count += 1
                        print(f"Image {img_path} has width {width} or height {height} less than 256")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    print(f"Total number of images with width or height less than 256 is {count}")


if __name__ == "__main__":
    folder_path = r"C:\Users\admin\Desktop\Flickr1024_Middlebury\train\input\left"
    check_images_folder(folder_path)
