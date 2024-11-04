# Лабораторная работа 1 по дисциплине МРЗвИС
# Выполнена студентом группы 121703 БГУИР Кимстач Д.Б.
# Реализация линейной рециркуляционной сети с постоянным коэффициентом обучения с нормированными весами
# как модели самокодировщика для задачи понижения размерности данных
# Вариант 11

# Ссылки на источники:
# https://numpy.org/doc/2.1/reference/index.html
# https://matplotlib.org/stable/api/index
# https://habr.com/ru/articles/130581/

import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt


class ImageCompressor:
    HIDDEN_SIZE = 42
    MAX_ERROR = 3500.0
    LEARNING_RATE = 0.007

    @classmethod
    def split_into_blocks(cls, height, width):
        img = cls.load_image()
        blocks = []
        for i in range(height // cls.BLOCK_HEIGHT):
            for j in range(width // cls.BLOCK_WIDTH):
                block = [
                    img[i * cls.BLOCK_HEIGHT + y, j * cls.BLOCK_WIDTH + x, color]
                    for y in range(cls.BLOCK_HEIGHT)
                    for x in range(cls.BLOCK_WIDTH)
                    for color in range(3)
                ]
                blocks.append(block)
        return np.array(blocks)

    @classmethod
    def blocks_to_image_array(cls, blocks, height, width):
        image_array = []
        blocks_in_line = width // cls.BLOCK_WIDTH
        for i in range(height // cls.BLOCK_HEIGHT):
            for y in range(cls.BLOCK_HEIGHT):
                line = [
                    [
                        blocks[i * blocks_in_line + j, (y * cls.BLOCK_WIDTH * 3) + (x * 3) + color]
                        for color in range(3)
                    ]
                    for j in range(blocks_in_line)
                    for x in range(cls.BLOCK_WIDTH)
                ]
                image_array.append(line)
        return np.array(image_array)

    @classmethod
    def display_image(cls, img_array):
        scaled_image = 1.0 * (img_array + 1) / 2
        plt.axis('off')
        plt.imshow(scaled_image)
        plt.show()

    @classmethod
    def load_image(cls):
        image = mpimg.imread("home.png")
        return (2.0 * image) - 1.0

    @classmethod
    def get_image_dimensions(cls):
        img = cls.load_image()
        return img.shape[0], img.shape[1]

    @classmethod
    def initialize_layers(cls):
        layer1 = np.random.rand(cls.INPUT_SIZE, cls.HIDDEN_SIZE) * 2 - 1
        layer2 = layer1.T

        for weight_matrix1_row in range(len(layer1)):
            for weight_matrix1_column in range(len(layer1[weight_matrix1_row])):
                denominator1 = cls.mod_of_vector(layer1.T[weight_matrix1_column])
                layer1[weight_matrix1_row][weight_matrix1_column] /= denominator1

        for weight_matrix2_row in range(len(layer2)):
            for weight_matrix2_column in range(len(layer2[weight_matrix2_row])):
                denominator2 = cls.mod_of_vector(layer2.T[weight_matrix2_column])
                layer2[weight_matrix2_row][weight_matrix2_column] /= denominator2

        return layer1, layer2

    @classmethod
    def train_model(cls):
        error = cls.MAX_ERROR + 1
        epoch = 0

        layer1, layer2 = cls.initialize_layers()

        while error > cls.MAX_ERROR:
            error = 0
            epoch += 1
            for block in cls.generate_blocks():
                hidden_layer = block @ layer1
                output_layer = hidden_layer @ layer2
                diff = output_layer - block
                layer1 -= cls.LEARNING_RATE * np.matmul(block.T, diff) @ layer2.T
                layer2 -= cls.LEARNING_RATE * hidden_layer.T @ diff

                for weight_matrix1_row in range(len(layer1)):
                    for weight_matrix1_column in range(len(layer1[weight_matrix1_row])):
                        denominator1 = cls.mod_of_vector(layer1.T[weight_matrix1_column])
                        layer1[weight_matrix1_row][weight_matrix1_column] /= denominator1

                for weight_matrix2_row in range(len(layer2)):
                    for weight_matrix2_column in range(len(layer2[weight_matrix2_row])):
                        denominator2 = cls.mod_of_vector(layer2.T[weight_matrix2_column])
                        layer2[weight_matrix2_row][weight_matrix2_column] /= denominator2

            error = sum(((block @ layer1 @ layer2 - block) ** 2).sum() for block in cls.generate_blocks())
            print(f'Epoch {epoch} - Error: {error}')

        compression_ratio = (cls.INPUT_SIZE * cls.total_blocks()) / (
                    (cls.INPUT_SIZE + cls.total_blocks()) * cls.HIDDEN_SIZE + 2)
        print(f'Compression Ratio: {compression_ratio}')
        return layer1, layer2

    @classmethod
    def mod_of_vector(cls, vector):
        return np.sqrt(np.sum(vector ** 2))

    @classmethod
    def generate_blocks(cls):
        height, width = cls.get_image_dimensions()
        return cls.split_into_blocks(height, width).reshape(cls.total_blocks(), 1, cls.INPUT_SIZE)

    @classmethod
    def total_blocks(cls):
        height, width = cls.get_image_dimensions()
        return (height * width) // (cls.BLOCK_HEIGHT * cls.BLOCK_WIDTH)

    @classmethod
    def compress_image(cls, block_height, block_width):

        cls.BLOCK_HEIGHT, cls.BLOCK_WIDTH = block_height, block_width
        cls.INPUT_SIZE = cls.BLOCK_HEIGHT * cls.BLOCK_WIDTH * 3

        height, width = cls.get_image_dimensions()
        layer1, layer2 = cls.train_model()

        original_image = cls.load_image()
        compressed_blocks = [block @ layer1 @ layer2 for block in cls.generate_blocks()]
        compressed_image = np.clip(np.array(compressed_blocks).reshape(cls.total_blocks(), cls.INPUT_SIZE), -1, 1)

        cls.display_image(original_image)
        cls.display_image(cls.blocks_to_image_array(compressed_image, height, width))


if __name__ == '__main__':
    ImageCompressor.compress_image(4, 4)
