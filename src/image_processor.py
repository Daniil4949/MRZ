# Лабораторная работа 1 по дисциплине МРЗвИС
# Выполнена студентом группы 121702 БГУИР Кимстач Д.Б.
# Реализация линейной рециркуляционной сети с постоянным коэффициентом обучения с нормированными весами
# Вариант 11
# Ссылки на источники:
# https://numpy.org/doc/2.1/reference/index.html
# https://matplotlib.org/stable/api/index
# https://habr.com/ru/articles/130581/
# https://studfile.net/preview/1557061/page:4/#9

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class ImageProcessor:
    @staticmethod
    def load_image(path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return (2.0 * image) - 1.0

    @staticmethod
    def split_into_blocks(image, block_height, block_width):
        height, width, _ = image.shape
        blocks = []
        for i in range(height // block_height):
            for j in range(width // block_width):
                block = image[
                    block_height * i : block_height * (i + 1),
                    block_width * j : block_width * (j + 1),
                    :3,
                ]
                blocks.append(block)
        return np.array(blocks)

    @staticmethod
    def blocks_to_image_array(blocks, height, width, block_height, block_width):
        image_array = []
        blocks_in_line = width // block_width
        for i in range(height // block_height):
            for y in range(block_height):
                line = [
                    [
                        blocks[
                            i * blocks_in_line + j,
                            (y * block_width * 3) + (x * 3) + color,
                        ]
                        for color in range(3)
                    ]
                    for j in range(blocks_in_line)
                    for x in range(block_width)
                ]
                image_array.append(line)
        return np.array(image_array)

    def get_image_dimensions(self):
        img = self.load_image("images/mountains.bmp")
        return img.shape[0], img.shape[1]

    def total_blocks(self, block_height, block_width):
        height, width = self.get_image_dimensions()
        return (height * width) // (block_height * block_width)

    @staticmethod
    def display_image(image_array):
        scaled_image = (image_array + 1) / 2
        plt.axis("off")
        plt.imshow(scaled_image)
        plt.show()

    @staticmethod
    def save_image(image_array, file_path="output.bmp"):
        image_uint8 = ((image_array + 1) * 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image_uint8, "RGB")
        image.save(file_path, format="BMP")
        print(f"Изображение сохранено в {file_path}")