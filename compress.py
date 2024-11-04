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


class ImageLoader:
    @staticmethod
    def load_image(filename="mountains.png"):
        image = mpimg.imread(filename)
        return (2.0 * image) - 1.0

    @staticmethod
    def display_image(img_array):
        scaled_image = 1.0 * (img_array + 1) / 2
        plt.axis('off')
        plt.imshow(scaled_image)
        plt.show()


class ImageCompressor:

    def __init__(self):
        self.hidden_size = 50
        self.max_error = 4000.0
        self.learning_rate = 0.017
        self.block_height = 4
        self.block_width = 4
        self.input_size = self.block_height * self.block_width * 3
        self.loader = ImageLoader()

    def split_into_blocks(self, height, width):
        img = self.loader.load_image()
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

    @staticmethod
    def load_image():
        image = mpimg.imread("mountains.png")
        return (2.0 * image) - 1.0

    def get_image_dimensions(self):
        img = self.loader.load_image()
        return img.shape[0], img.shape[1]

    def initialize_layers(self):
        layer1 = self.initialize_weights(self.input_size, self.hidden_size)
        layer2 = layer1.T
        layer1 = self.normalize_layer(layer1)
        layer2 = self.normalize_layer(layer2)
        return layer1, layer2

    @staticmethod
    def initialize_weights(input_size, hidden_size):
        return np.random.rand(input_size, hidden_size) * 2 - 1

    def normalize_layer(self, layer):
        for weight_matrix_row in range(len(layer)):
            for weight_matrix_column in range(len(layer[weight_matrix_row])):
                denominator = self.mod_of_vector(layer.T[weight_matrix_column])
                layer[weight_matrix_row][weight_matrix_column] /= denominator
        return layer

    def train_model(self):
        error = self.max_error + 1
        epoch = 0

        layer1, layer2 = self.initialize_layers()

        while error > self.max_error:
            error = self.train_epoch(layer1, layer2)
            epoch += 1
            print(f'Epoch {epoch} - Error: {error}')

        compression_ratio = self.calculate_compression_ratio()
        print(f'Compression Ratio: {compression_ratio}')
        return layer1, layer2

    def train_epoch(self, layer1, layer2):
        """Обучение модели на одной эпохе."""
        for block in self.generate_blocks():
            hidden_layer = block @ layer1
            output_layer = hidden_layer @ layer2
            diff = output_layer - block
            layer1 -= self.learning_rate * np.matmul(block.T, diff) @ layer2.T
            layer2 -= self.learning_rate * hidden_layer.T @ diff

            self.normalize_weights(layer1)
            self.normalize_weights(layer2)

        error = self.calculate_error(layer1, layer2)
        return error

    def normalize_weights(self, layer):
        for weight_matrix_row in range(len(layer)):
            for weight_matrix_column in range(len(layer[weight_matrix_row])):
                denominator = self.mod_of_vector(layer.T[weight_matrix_column])
                layer[weight_matrix_row][weight_matrix_column] /= denominator

    def calculate_error(self, layer1, layer2):
        return sum(((block @ layer1 @ layer2 - block) ** 2).sum() for block in self.generate_blocks())

    def calculate_compression_ratio(self):
        return (self.input_size * self.total_blocks()) / (
                (self.input_size + self.total_blocks()) * self.hidden_size + 2)

    @staticmethod
    def mod_of_vector(vector):
        return np.sqrt(np.sum(vector ** 2))

    def generate_blocks(self):
        height, width = self.get_image_dimensions()
        return self.split_into_blocks(height, width).reshape(self.total_blocks(), 1, self.input_size)

    def total_blocks(self):
        height, width = self.get_image_dimensions()
        return (height * width) // (self.block_height * self.block_width)

    def compress_image(self):
        layer1, layer2 = self.train_model()

        original_image = self.loader.load_image()
        compressed_image = self.compress_and_reformat_blocks(layer1, layer2)

        self.loader.display_image(original_image)
        height, width = self.get_image_dimensions()
        self.loader.display_image(self.blocks_to_image_array(compressed_image, height, width))

    def configure_block_parameters(self, block_height, block_width):
        self.block_height = block_height
        self.block_width = block_width
        self.input_size = self.block_height * self.block_width * 3

    def compress_and_reformat_blocks(self, layer1, layer2):
        compressed_blocks = [block @ layer1 @ layer2 for block in self.generate_blocks()]
        compressed_image = np.clip(np.array(compressed_blocks).reshape(self.total_blocks(), self.input_size), -1, 1)
        return compressed_image

    def display_images(self, original_image, compressed_image):
        self.display_image(original_image)
        height, width = self.get_image_dimensions()
        self.display_image(self.blocks_to_image_array(compressed_image, height, width))


if __name__ == '__main__':
    ImageCompressor().compress_image()
