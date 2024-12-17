# Лабораторная работа 1 по дисциплине МРЗвИС
# Выполнена студентом группы 121702 БГУИР Кимстач Д.Б.
# Реализация линейной рециркуляционной сети с постоянным коэффициентом обучения с нормированными весами
# Вариант 11
# Ссылки на источники:
# https://numpy.org/doc/2.1/reference/index.html
# https://matplotlib.org/stable/api/index
# https://habr.com/ru/articles/130581/
# https://studfile.net/preview/1557061/page:4/#9

# import matplotlib.pyplot as plt
# import numpy as np
#
# parameters = [
#     0.00001,
#     0.00002,
#     0.00003,
#
#     0.00004,
#     0.00005,
#     0.00006,
#     0.0001,
#     0.0002,
#     0.0003,
#
#     0.0004,
#     0.0005,
#     0.0006,
#     0.0007,
#     0.0008,
#     0.0009,
#     0.001,
#     0.002,
#     0.002,
#     0.0021,
#     0.0022,
#     0.0023,
# ]
# values = [
#     [563, 550, 557],
#     [244, 247, 248],
#     [201, 199, 205], [139, 142, 138],
#     [114, 111, 115],
#
#     [80, 85, 83],
#     [55, 63, 58],
#     [28, 31, 32], [19, 20, 22], [13, 12, 14]
#
#     ,
#     [12, 12, 11], [10, 10, 11], [9, 8, 9], [7, 8, 8], [7, 7, 8], [7, 7,6], [7, 7,6], [10, 12, 11],  # для 0.002
#     [11, 13, 12],  # для 0.0021
#     [12, 14, 13],  # для 0.0022
#     [13, 15, 14],  # для 0.0023
#
# ]
#
# averages = [np.mean(group) for group in values]
#
# plt.figure(figsize=(10, 6))
#
# for param, group in zip(parameters, values):
#     plt.scatter([param] * len(group), group, color='blue',
#                 label='Количество итераций' if param == parameters[0] else "")
#
# plt.plot(parameters, averages, color='red', linestyle='-', marker='o', label='Среднее количество итераций')
#
# plt.title("График зависимости количества итераций обучения от коэффициента обучения")
# plt.xlabel("Коэффициент обучения")
# plt.ylabel("Количество итераций обучения")
# plt.legend()
# plt.grid(True)
#
# plt.axvline(x=0.0024, color='black', linestyle='--', label="Разрыв")
#
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

parameters = [
    0.0001,
    0.0002,
    0.0003,
    0.0004,
    0.0005,
    0.0006,
    0.0007,
    0.0008,
    0.0009,
    0.001,
    0.002,
    0.0021,
    0.0022,
    0.0023,
]

# Значения, где будет происходить "бесконечность"
values = [
    [55, 63, 58],
    [28, 31, 32], [19, 20, 22], [13, 12, 14],
    [12, 12, 11], [10, 10, 11], [9, 8, 9],
    [7, 7, 8], [7, 7, 6], [7, 7, 6],
    [10, 12, 11],  # для 0.002
    [11, 13, 12],  # для 0.0021
    [12, 14, 13],  # для 0.0022
    [13, 15, 14],  # для 0.0023
]

# Рассчитываем среднее, игнорируя бесконечности
averages = [
    np.mean(group) if not np.all(np.isinf(group)) else np.nan  # Используем NaN для разрыва
    for group in values
]

# Построение графика
plt.figure(figsize=(10, 6))

# Для отображения точек
for param, group in zip(parameters, values):
    group_to_plot = [1e6 if x == np.inf else x for x in group]  # Заменяем бесконечность на большое число
    plt.scatter([param] * len(group), group_to_plot, color='blue',
                label='Количество итераций' if param == parameters[0] else "")

# Построение средней линии, пропуская NaN
plt.plot(parameters, averages, color='red', linestyle='-', marker='o', label='Среднее количество итераций')

# Добавим вертикальные линии для обозначения разрыва
plt.axvline(x=0.0024, color='black', linestyle='--', label="Разрыв")

plt.title("График зависимости количества итераций обучения от коэффициента обучения")
plt.xlabel("Коэффициент обучения")
plt.ylabel("Количество итераций обучения")
plt.legend()
plt.grid(True)

# Добавляем дополнительные деления на оси X
x_ticks = np.arange(0.0001, 0.0024, 0.0005)  # создаем деления от 0.0001 до 0.0023 с шагом 0.0001
plt.xticks(np.concatenate([x_ticks, [0.0024]]))  # добавляем точки для 0.0024 и 0.0025

# Показываем график
plt.show()
