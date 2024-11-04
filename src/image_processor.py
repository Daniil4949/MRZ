# Лабораторная работа 1 по дисциплине МРЗвИС
# Выполнена студентом группы 121703 БГУИР Кимстач Д.Б.
# Реализация линейной рециркуляционной сети с постоянным коэффициентом обучения с нормированными весами
# как модели самокодировщика для задачи понижения размерности данных
# Вариант 11

# Ссылки на источники:
# https://numpy.org/doc/2.1/reference/index.html
# https://matplotlib.org/stable/api/index
# https://habr.com/ru/articles/130581/

import numpy as np
from matplotlib import pyplot as plt


class ImageProcessor:
    def __init__(self, block_height=4, block_width=4):
        self.block_height = block_height
        self.block_width = block_width

    def split_into_blocks(self, img):
        height, width = img.shape[:2]
        blocks = []
        for i in range(height // self.block_height):
            for j in range(width // self.block_width):
                block = [
                    img[i * self.block_height + y, j * self.block_width + x, color]
                    for y in range(self.block_height)
                    for x in range(self.block_width)
                    for color in range(3)
                ]
                blocks.append(block)
        return np.array(blocks)

    def blocks_to_image_array(self, blocks, height, width):
        image_array = []
        blocks_in_line = width // self.block_width
        for i in range(height // self.block_height):
            for y in range(self.block_height):
                line = [
                    [
                        blocks[i * blocks_in_line + j, (y * self.block_width * 3) + (x * 3) + color]
                        for color in range(3)
                    ]
                    for j in range(blocks_in_line)
                    for x in range(self.block_width)
                ]
                image_array.append(line)
        return np.array(image_array)

    @staticmethod
    def display_image(img_array):
        scaled_image = 1.0 * (img_array + 1) / 2
        plt.axis('off')
        plt.imshow(scaled_image)
        plt.show()
