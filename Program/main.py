import re
import joblib
import pymystem3

# Добавляем переменные, которые использовали ранее, чтобы использовать их снова
vectorizer = joblib.load('vectorizer.pkl') # Нам нужен не новый, а от, что был изначально
mystem = pymystem3.Mystem()
regex = re.compile('[^a-zA-Z]')

# Загружаем модеель
model = joblib.load("catastrophe_model.pkl")

# Цикл запросов
input_string = input("Введите предложение для анализа: ")
while input_string != '0':
    # Преобразования в строке
    input_string = regex.sub('', input_string.lower())
    input_string = ''.join((mystem.lemmatize(input_string)))

    # Слова в числа
    input_string_features = vectorizer.transform([input_string])

    # Предсказание
    prediction = model.predict(input_string_features)[0]

    # Вывод результатов
    print('Существует верояность катастрофы' if prediction == 1 else 'Вероятность катастрофы не обнаружена')

    # Ввод очередной строки
    input_string = input("Если хотите закончить: введите 0\nЕсли вы хотите продолжить: введите предложение для "
                         "анализа: ")


