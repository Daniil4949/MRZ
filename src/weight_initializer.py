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
import tensorflow as tf


class WeightInitializer:
    @staticmethod
    def initialize_layers(input_size, hidden_size):
        # Функция `tf.random.uniform` обладает равномерным распределением
        first_layer = tf.random.uniform((input_size, hidden_size)) * 2 - 1
        second_layer = tf.transpose(first_layer)

        first_layer = WeightInitializer.normalize(first_layer)
        second_layer = WeightInitializer.normalize(second_layer)

        return first_layer, second_layer

    @staticmethod
    def normalize(weights):
        weights = weights.numpy()
        denominator = np.sqrt(np.sum(weights**2, axis=0))

        for col_index in range(weights.shape[1]):
            if denominator[col_index] == 0:
                weights[:, col_index] = 0
            else:
                weights[:, col_index] /= denominator[col_index]

        return tf.convert_to_tensor(weights)
