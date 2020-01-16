#coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os

# Функция отвечает за подготовку данных из репозитория.
# Результатом работы являются два CSV-файла с подогнанными под Tensor Flow данными для обучения и тестирования нейронной сети. В частности, категорийные параметры грибов конвертируются в числовые (0 и 1)
def prepare_data(data_file_name):
    header = ['<DATE>','<TIME>','<OPEN>','<HIGH>','<LOW>','<CLOSE>','<VOL>']
    df = pd.read_csv(data_file_name, sep=',', names=header)

    # Теперь надо разделить конвертированные данные
    # на два набора - один для тренировки (большой)
    # и один для тестирования нейросети (поменьше)
    df_train, df_test = train_test_split(df, test_size=0.1)

    # Определяем количество строк и столбцов в каждом из наборов данных
    num_train_entries = df_train.shape[0]
    num_train_features = df_train.shape[1] - 1

    num_test_entries = df_test.shape[0]
    num_test_features = df_test.shape[1] - 1

    # Итоговые наборы записываем во временные csv-файлы, т.к.
    # необходимо записать количества столбцов и строк в начало шапки
    # рабочих csv, как того требует Tensor Flow
    df_train.to_csv('train_temp.csv', index=False)
    df_test.to_csv('test_temp.csv', index=False)

    # Пишем количества в тренировочный файл, затем в тестовый
    with open("mushroom_train.csv", "w") as mushroom_train:
        with open("train_temp.csv") as train_temp:
            mushroom_train.write(str(num_train_entries) +
                                 "," + str(num_train_features) +
                                 "," + train_temp.read())

    with open("mushroom_test.csv", "w") as mushroom_test:
        with open("test_temp.csv") as test_temp:
            mushroom_test.write(str(num_test_entries) +
                                "," + str(num_test_features) +
                                "," + test_temp.read())

    # Удаляем временные файлы, они больше не нужны
    os.remove("train_temp.csv")
    os.remove("test_temp.csv")

# Функция формирует входные данные для тестирования для Tensor Flow
def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y

# Функция формирует входные данные для тренировки для Tensor Flow
def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y

# Функция возвращает данные двух пробных грибов для
# предсказания их съедобности (ожидаемый результат: съедобен, ядовит)
# Иными словами, это функция для проверки обученной и протестированной нейросети
def new_samples():
    return np.array([
            [20200116,190000,2718.2000000,2722.4000000,2714.0000000,2719.8000000,40921],
            [20200116,180000,2762.0000000,2762.8000000,2727.4000000,2739.0000000,197024]], dtype=np.int)

if __name__ == "__main__":
    MUSHROOM_DATA_FILE = "YNDX_150101_200116.data"

# Подготавливаем данные грибов для Tensor Flow,
# создав два CSV-файла (тренировка и тест)
prepare_data(MUSHROOM_DATA_FILE)

# Загружаем подготовленные данные
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename='mushroom_train.csv',
    target_dtype=np.int,
    features_dtype=np.int,
    target_column=0)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename='mushroom_test.csv',
    target_dtype=np.int,
    features_dtype=np.int,
    target_column=0)

# Определяем, что все параметры цветов имеют реальные значения (подробнее ниже)
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=6)]

# Создаем трехслойную DNN-нейросеть с 10, 20 и 10 нейронами в слое
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 20, 10],
    n_classes=2,
    model_dir="/tmp/mushroom_model")

# Тренируем нейросеть
classifier.fit(input_fn=get_train_inputs, steps=2000)

# Нормализуем нейросеть с помощью тестового набора данных
accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                     steps=1)["accuracy"]

print("\nТочность предсказаний: {0:f}\n".format(accuracy_score))

# Пробуем запустить нейросеть на двух наших пробных грибах
predictions = list(classifier.predict_classes(input_fn=new_samples))

print("Предсказания съедобности пробных грибов: {}\n"
      .format(predictions))