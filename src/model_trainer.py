# Лабораторная работа 1 по дисциплине МРЗвИС
# Выполнена студентом группы 121703 БГУИР Кимстач Д.Б.
# Реализация линейной рециркуляционной сети с постоянным коэффициентом обучения с нормированными весами
# Вариант 11
# Ссылки на источники:
# https://numpy.org/doc/2.1/reference/index.html
# https://matplotlib.org/stable/api/index
# https://habr.com/ru/articles/130581/
# https://studfile.net/preview/1557061/page:4/#9


import tensorflow as tf

from src.weight_initializer import WeightInitializer


class ModelTrainer:
    def __init__(self, input_size, hidden_size, learning_rate, max_error):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.max_error = max_error

    def train_model(self, blocks):
        first_layer, second_layer = WeightInitializer.initialize_layers(
            self.input_size, self.hidden_size
        )
        error = self.max_error + 1
        epoch = 0

        while error > self.max_error:
            epoch += 1
            for block in blocks:
                hidden_layer = tf.matmul(block, first_layer)
                output_layer = tf.matmul(hidden_layer, second_layer)
                diff = output_layer - block

                first_layer -= self.learning_rate * tf.matmul(
                    tf.matmul(tf.transpose(block), diff), tf.transpose(second_layer)
                )
                second_layer -= self.learning_rate * tf.matmul(
                    tf.transpose(hidden_layer), diff
                )

                first_layer = WeightInitializer.normalize(first_layer)
                second_layer = WeightInitializer.normalize(second_layer)

            error = sum(
                tf.reduce_sum((block @ first_layer @ second_layer - block) ** 2)
                for block in blocks
            )
            print(f"Epoch {epoch} - Error: {error}")

        return first_layer, second_layer
