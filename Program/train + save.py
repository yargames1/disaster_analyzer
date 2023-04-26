# Импорт необходимых библиотек
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import re
import joblib
import pymystem3
import time

pd.options.mode.chained_assignment = None  # default='warn'  -  если не поставить, будет постоянное предупреждение об
# изменениях в text_data и labels

mystem = pymystem3.Mystem()
regex = re.compile('[^a-zA-Z]')

# Загрузка и предобработка данных
data = pd.read_csv("train_full.csv")  # Получаем таблицу для работы


# Преобразуем сторки
def lemm(s):
    return ''.join((mystem.lemmatize(s)))


for i in range(len(data)):
    data['text'][i] = regex.sub('', data['text'][i].lower())

text_data = data["text"]  # Забираем коментарии
labels = data["target"]  # Забираем результаты

text_data = text_data.apply(lemm)

# Слова в числа
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_data)

# Обучение модели
clf = SGDClassifier()  # Используем SGDClassifier в качестве модели
clf.fit(X[:900], labels[:900])  # Обучение модели на почти всех доступных данных

# Проверим точность модели
right_answers = 0
for i in range(0, 100):
    if clf.predict(X[900 + i])[0] == labels[900 + i]:
        right_answers += 1
print("Точность модели:", str(right_answers / 100))

# Сохраняем модель
joblib.dump(clf, "catastrophe_model.pkl")
joblib.dump(vectorizer, 'vectorizer.pkl') # Оказывается, это тоже очень нужно
print('Модель сохранена')
# Делаем так, чтобы можно было узнать, что произошло
time.sleep(60)

