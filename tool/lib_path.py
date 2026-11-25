# License: AGPL-3.0

"""
给定一个输入文件夹
给定一个输出文件夹
将输入文件夹中的所有图像文件进行处理，并将结果保存到输出文件夹

保持原本的目录结构
"""

import pathlib
import os



def process_images(input_folder: pathlib.Path, output_folder: pathlib.Path):
    """
    处理输入文件夹中的所有图像文件，并将结果保存到输出文件夹中，保持原本的目录结构
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 遍历输入文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # 检查文件是否为图像文件（根据扩展名判断）
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # 构建输入图像的完整路径
                input_image_path = pathlib.Path(root) / file

                # 构建输出图像的完整路径，保持原本的目录结构
                relative_path = pathlib.Path(root).relative_to(input_folder)
                output_image_path = output_folder / relative_path / file

                # 创建输出图像所在的目录（如果不存在）
                output_image_path.parent.mkdir(parents=True, exist_ok=True)

                # 在这里添加对图像的处理逻辑，例如读取、处理和保存图像
                # 例如：
                # image = read_image(input_image_path)
                # processed_image = process_image(image)
                # save_image(processed_image, output_image_path)

                print(f"Processed {input_image_path} -> {output_image_path}")