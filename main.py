# Лабораторная работа 1 по дисциплине МРЗвИС
# Выполнена студентом группы 121703 БГУИР Кимстач Д.Б.
# Реализация линейной рециркуляционной сети с постоянным коэффициентом обучения с нормированными весами
# Вариант 11
# Ссылки на источники:
# https://numpy.org/doc/2.1/reference/index.html
# https://matplotlib.org/stable/api/index
# https://habr.com/ru/articles/130581/
# https://studfile.net/preview/1557061/page:4/#9

from src.image_compressor import ImageCompressor

if __name__ == "__main__":
    compressor = ImageCompressor(
        image_path="mountains.bmp",
        block_height=8,
        block_width=8,
        hidden_size=45,
        learning_rate=0.0015,
        max_error=2500.0,
    )
    compressor.compress_image()
