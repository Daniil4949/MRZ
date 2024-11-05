# Лабораторная работа 1 по дисциплине МРЗвИС
# Выполнена студентом группы 121703 БГУИР Кимстач Д.Б.
# Реализация линейной рециркуляционной сети с постоянным коэффициентом обучения с нормированными весами
# Вариант 11

# Ссылки на источники:
# https://numpy.org/doc/2.1/reference/index.html
# https://matplotlib.org/stable/api/index
# https://habr.com/ru/articles/130581/
# https://studfile.net/preview/1557061/page:4/#9

import numpy as np

from src.image_loader import ImageLoader
from src.image_processor import ImageProcessor


class ImageCompressor:
    def __init__(self):
        self.hidden_size = 45
        self.max_error = 4000.0
        self.learning_rate = 0.017
        self.loader = ImageLoader()
        self.processor = ImageProcessor()

        # Initialize dimensions based on default block size
        self.block_height = self.processor.block_height
        self.block_width = self.processor.block_width
        self.input_size = self.block_height * self.block_width * 3

    def get_image_dimensions(self):
        img = self.loader.load_image()
        return img.shape[0], img.shape[1]

    def initialize_layers(self):
        first_layer = self.initialize_weights(self.input_size, self.hidden_size)
        second_layer = first_layer.T
        first_layer = self.normalize_layer(first_layer)
        second_layer = self.normalize_layer(second_layer)
        return first_layer, second_layer

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
        step = 0

        first_layer, second_layer = self.initialize_layers()

        while error > self.max_error:
            error = self.train_epoch(first_layer, second_layer)
            step += 1
            print(f'Step {step} - Error rate: {error}')

        compression_ratio = self.calculate_compression_ratio()
        print(f'Compression value: {compression_ratio}')
        return first_layer, second_layer

    def train_epoch(self, first_layer, second_layer):
        for block in self.generate_blocks():
            hidden_layer = block @ first_layer
            output_layer = hidden_layer @ second_layer
            diff = output_layer - block
            first_layer -= self.learning_rate * np.matmul(block.T, diff) @ second_layer.T
            second_layer -= self.learning_rate * hidden_layer.T @ diff

            self.normalize_weights(first_layer)
            self.normalize_weights(second_layer)

        error = self.calculate_error(first_layer, second_layer)
        return error

    def normalize_weights(self, layer):
        for weight_matrix_row in range(len(layer)):
            for weight_matrix_column in range(len(layer[weight_matrix_row])):
                denominator = self.mod_of_vector(layer.T[weight_matrix_column])
                layer[weight_matrix_row][weight_matrix_column] /= denominator

    def calculate_error(self, first_layer, second_layer):
        return sum(((block @ first_layer @ second_layer - block) ** 2).sum() for block in self.generate_blocks())

    def calculate_compression_ratio(self):
        return (self.input_size * self.total_blocks()) / (
                (self.input_size + self.total_blocks()) * self.hidden_size + 2)

    @staticmethod
    def mod_of_vector(vector):
        return np.sqrt(np.sum(vector ** 2))

    def generate_blocks(self):
        img = self.loader.load_image()
        return self.processor.split_into_blocks(img).reshape(self.total_blocks(), 1, self.input_size)

    def total_blocks(self):
        height, width = self.get_image_dimensions()
        return (height * width) // (self.block_height * self.block_width)

    def compress_image(self):
        first_layer, second_layer = self.train_model()

        original_image = self.loader.load_image()
        compressed_image = self.compress_and_reformat_blocks(first_layer, second_layer)

        self.processor.display_image(original_image)
        height, width = self.get_image_dimensions()
        self.processor.display_image(self.processor.blocks_to_image_array(compressed_image, height, width))

    def compress_and_reformat_blocks(self, first_layer, second_layer):
        compressed_blocks = [block @ first_layer @ second_layer for block in self.generate_blocks()]
        compressed_image = np.clip(np.array(compressed_blocks).reshape(self.total_blocks(), self.input_size), -1, 1)
        return compressed_image
