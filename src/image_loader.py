# Лабораторная работа 1 по дисциплине МРЗвИС
# Выполнена студентом группы 121703 БГУИР Кимстач Д.Б.
# Реализация линейной рециркуляционной сети с постоянным коэффициентом обучения с нормированными весами
# как модели самокодировщика для задачи понижения размерности данных
# Вариант 11

# Ссылки на источники:
# https://numpy.org/doc/2.1/reference/index.html
# https://matplotlib.org/stable/api/index
# https://habr.com/ru/articles/130581/


from matplotlib import image as mpimg


class ImageLoader:
    @staticmethod
    def load_image(filename="mountains.png"):
        image = mpimg.imread(filename)
        return (2.0 * image) - 1.0
