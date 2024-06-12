
# Общее

Делаем все в одной монорепе.

1) Данные - параллелится

2) Модель - параллелится
Принимает конфиг - там описывается, откуда берет даные

3) Инфра по запускук экспов - параллелится, если определить интерфейсы для модели

4) Запуск экспериментов и разработка алгоритмов для матчинга

# Глоссарий
* Эмбэддинг -- это вектор, который характеризует отрывок видео/аудио. Его генерит нейронка.
* Фингерпринт -- отпечаток аудио/видео, представляет из себя упорядоченную последовательность эмбэддингов

# QA
* Нужно ли матчить все старые UGC если добавляется новое видео в индекс?
* Какое соотношение в количестве лицензионного и UGC контента?

# Требования

# Требования для матчинга аудио
* минимальная длинна сметченного отрезка - 15 секунд
* Независимость от смещения на воспроизведение
* Независимость от изменения разрешения видео
* Независимость от изменения громкости аудио

# Требования для матчинга видео
* Матчим ли мы отрезки? Или матчим только полное видео?
* независимость к добавлению шума
* независимость к небольшому смещению

TODO
* ускорение/замедление
* независимость от рандомных редких кадров
* независимость от вотер-марок
* независимость тот различия частоты аудио

# Алгоритм для матчинга видео

* можем отфильтровать видео/аудио, которые были загружены до лицензионного
* алгоритм определения опорных кадров + длительность для опорных кадров
* получение эмбэддингов для опорных кадров

TODO отсечение похожих опорных кадров в рамках одного видео

# Поиск по сматченному
* ищем полное совпадение
* ищем ближайших соседей + пересечения по длительности

# Ютуб

https://support.google.com/youtube/answer/2807622?sjid=15232144134161570872-EU

Определяет только полностью или почти похожие копии. Маленькие кусочки не умееет определять.


# Отчет

Сюда будем фиксировать структуру отчета и экспериментов, которые мы будем питчить.

Можно записывать вопросы, на которые хочется ответить в результате работы.

* Как влияет размерность после применения PCA на качество?
* Как повлияет квантизация?
* Как влияет добавление последовательности векторов? Вместо одного вектора на всю аудюшку?

* Исследовать или погуглить про advversarial атаки на такие системы матчинга -- как от них зачититься?


Рассказать, что мы подумали про удаление слепков из индекса.

## Расчет производительности и необходимого затраченого времени:

Скорость матчинга по аудио:
* x60 от realtime скорости
* если в сутки заливается 5к часов видео, потребуется ~84 часа на вычисление слепков
* слепки одного часа видео занимают 6Мb
* в памяти индекс рутуба весит 7 GB с учетом индекса

* выйграем ли мы от каскадирования?
