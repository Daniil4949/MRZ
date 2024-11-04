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
    HIDDEN_SIZE = 50
    MAX_ERROR = 4000.0
    LEARNING_RATE = 0.017

    def split_into_blocks(self, height, width):
        img = self.load_image()
        blocks = []
        for i in range(height // self.BLOCK_HEIGHT):
            for j in range(width // self.BLOCK_WIDTH):
                block = [
                    img[i * self.BLOCK_HEIGHT + y, j * self.BLOCK_WIDTH + x, color]
                    for y in range(self.BLOCK_HEIGHT)
                    for x in range(self.BLOCK_WIDTH)
                    for color in range(3)
                ]
                blocks.append(block)
        return np.array(blocks)

    def blocks_to_image_array(self, blocks, height, width):
        image_array = []
        blocks_in_line = width // self.BLOCK_WIDTH
        for i in range(height // self.BLOCK_HEIGHT):
            for y in range(self.BLOCK_HEIGHT):
                line = [
                    [
                        blocks[i * blocks_in_line + j, (y * self.BLOCK_WIDTH * 3) + (x * 3) + color]
                        for color in range(3)
                    ]
                    for j in range(blocks_in_line)
                    for x in range(self.BLOCK_WIDTH)
                ]
                image_array.append(line)
        return np.array(image_array)

    @staticmethod
    def display_image(img_array):
        scaled_image = 1.0 * (img_array + 1) / 2
        plt.axis('off')
        plt.imshow(scaled_image)
        plt.show()

    @staticmethod
    def load_image():
        image = mpimg.imread("mountains.png")
        return (2.0 * image) - 1.0

    def get_image_dimensions(self):
        img = self.load_image()
        return img.shape[0], img.shape[1]

    def initialize_layers(self):
        layer1 = self.initialize_weights(self.INPUT_SIZE, self.HIDDEN_SIZE)
        layer2 = layer1.T
        layer1 = self.normalize_layer(layer1)
        layer2 = self.normalize_layer(layer2)
        return layer1, layer2

    def initialize_weights(self, input_size, hidden_size):
        return np.random.rand(input_size, hidden_size) * 2 - 1

    def normalize_layer(self, layer):
        for weight_matrix_row in range(len(layer)):
            for weight_matrix_column in range(len(layer[weight_matrix_row])):
                denominator = self.mod_of_vector(layer.T[weight_matrix_column])
                layer[weight_matrix_row][weight_matrix_column] /= denominator
        return layer

    def train_model(self):
        error = self.MAX_ERROR + 1
        epoch = 0

        layer1, layer2 = self.initialize_layers()

        while error > self.MAX_ERROR:
            error = 0
            epoch += 1
            for block in self.generate_blocks():
                hidden_layer = block @ layer1
                output_layer = hidden_layer @ layer2
                diff = output_layer - block
                layer1 -= self.LEARNING_RATE * np.matmul(block.T, diff) @ layer2.T
                layer2 -= self.LEARNING_RATE * hidden_layer.T @ diff

                for weight_matrix1_row in range(len(layer1)):
                    for weight_matrix1_column in range(len(layer1[weight_matrix1_row])):
                        denominator1 = self.mod_of_vector(layer1.T[weight_matrix1_column])
                        layer1[weight_matrix1_row][weight_matrix1_column] /= denominator1

                for weight_matrix2_row in range(len(layer2)):
                    for weight_matrix2_column in range(len(layer2[weight_matrix2_row])):
                        denominator2 = self.mod_of_vector(layer2.T[weight_matrix2_column])
                        layer2[weight_matrix2_row][weight_matrix2_column] /= denominator2

            error = sum(((block @ layer1 @ layer2 - block) ** 2).sum() for block in self.generate_blocks())
            print(f'Epoch {epoch} - Error: {error}')

        compression_ratio = (self.INPUT_SIZE * self.total_blocks()) / (
                (self.INPUT_SIZE + self.total_blocks()) * self.HIDDEN_SIZE + 2)
        print(f'Compression Ratio: {compression_ratio}')
        return layer1, layer2

    @staticmethod
    def mod_of_vector(vector):
        return np.sqrt(np.sum(vector ** 2))

    def generate_blocks(self):
        height, width = self.get_image_dimensions()
        return self.split_into_blocks(height, width).reshape(self.total_blocks(), 1, self.INPUT_SIZE)

    def total_blocks(self):
        height, width = self.get_image_dimensions()
        return (height * width) // (self.BLOCK_HEIGHT * self.BLOCK_WIDTH)

    def compress_image(self, block_height, block_width):

        self.BLOCK_HEIGHT, self.BLOCK_WIDTH = block_height, block_width
        self.INPUT_SIZE = self.BLOCK_HEIGHT * self.BLOCK_WIDTH * 3

        height, width = self.get_image_dimensions()
        layer1, layer2 = self.train_model()

        original_image = self.load_image()
        compressed_blocks = [block @ layer1 @ layer2 for block in self.generate_blocks()]
        compressed_image = np.clip(np.array(compressed_blocks).reshape(self.total_blocks(), self.INPUT_SIZE), -1, 1)

        self.display_image(original_image)
        self.display_image(self.blocks_to_image_array(compressed_image, height, width))


if __name__ == '__main__':
    ImageCompressor().compress_image(4, 4)
