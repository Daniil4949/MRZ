# Лабораторная работа 1 по дисциплине МРЗвИС
# Выполнена студентом группы 121702 БГУИР Кимстач Д.Б.
# Реализация линейной рециркуляционной сети с постоянным коэффициентом обучения с нормированными весами
# Вариант 11
# Ссылки на источники:
# https://numpy.org/doc/2.1/reference/index.html
# https://matplotlib.org/stable/api/index
# https://habr.com/ru/articles/130581/
# https://studfile.net/preview/1557061/page:4/#9

import numpy as np

from src.image_processor import ImageProcessor
from src.model_trainer import ModelTrainer


class ImageCompressor:
    def __init__(
        self,
        image_path,
        block_height,
        block_width,
        hidden_size,
        learning_rate,
        max_error,
    ):
        self.image_path = image_path
        self.block_height = block_height
        self.block_width = block_width
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.max_error = max_error
        self.input_size = block_height * block_width * 3

    def compression_ratio(self, total_blocks):
        # Изменен расчет коэффициента сжатия
        return (32 * self.block_height * self.block_width * total_blocks) / (
            (self.input_size + total_blocks) * 32 * self.hidden_size + 2
        )

    def compress_image(self):
        image = ImageProcessor.load_image(self.image_path)
        height, width, _ = image.shape

        blocks = ImageProcessor.split_into_blocks(
            image, self.block_height, self.block_width
        )
        blocks = blocks.reshape(len(blocks), 1, self.input_size)

        trainer = ModelTrainer(
            self.input_size, self.hidden_size, self.learning_rate, self.max_error
        )
        first_layer, second_layer = trainer.train_model(blocks)

        compressed_blocks = [block @ first_layer @ second_layer for block in blocks]
        compressed_image = np.clip(
            np.array(compressed_blocks).reshape(
                len(compressed_blocks), self.input_size
            ),
            -1,
            1,
        )
        total_blocks = ImageProcessor().total_blocks(
            block_height=self.block_height, block_width=self.block_width
        )

        compression_ratio = self.compression_ratio(total_blocks)

        print(f"Compression ratio: {compression_ratio}")
        ImageProcessor.display_image(image)
        ImageProcessor.display_image(
            ImageProcessor.blocks_to_image_array(
                compressed_image, height, width, self.block_height, self.block_width
            )
        )
