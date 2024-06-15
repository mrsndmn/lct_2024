
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

# Какие процессы можно распараллелить? - нарисовать сиквенс диаграмму

Рассказать, что мы подумали про удаление слепков из индекса.

## Расчет производительности и необходимого затраченого времени:

Скорость матчинга по аудио:
* x60 от realtime скорости
* если в сутки заливается 5к часов видео, потребуется ~84 часа на вычисление слепков
* слепки одного часа видео занимают 6Мb
* в памяти индекс рутуба весит 7 GB с учетом индекса

* выйграем ли мы от каскадирования?

### Гипотезы

Вычисления для матчинга разбиваем на 4 этапа:
* нормализация
* предобработка
* вычисление контекстных эмбэддингов - запуск самого тела нейронки с ограниченным контекстом (основная вычислительная нагрузка)
* вычисление эмбэддингов интервалов - пуллинг интервалов с предыдущего слоя (дешево)

TODO возможно, понижение размерности или квантизация

## Генерация эмбэддингов:
* можно ускорить, если один раз предобрабатывать пересекающиеся интервалы для контекстонезависимых слоев

## Матчинг:

* Иногда попадаются рандомные интервалы - их можно отсеять с помощью эвристик
* Тк в пиратском видео данные могут быть с небольшим или большим оффсетом -- нужно генерить эмбэддинги с маленьким шагом и оптимизация пуллинга в этом нам помогает


## Поиск:
* Можно ли ускорить поиск используя dot вместо cosine?

Да! Нужно нормализовать векторы и делать поиск с порощью `DOT` расстояния на уже нормализованных
векторах. Значение метрики для нормализованных векторов получаются одинаковыми. А вычислений
нужно намного меньше.

Правильный выбор метрики ускоряет поиск по индексу

# TODO перезапустить бенчмарк с разными размерностями скрытого пространства
```
# тест-бенчмарк, как расстояние влияет
# на скорость работы векторного поиска
pytest -s src/avm/search/audio_test.py

metric=Dot      time= 0.010836975419997542
metric=Cosine   time= 0.10016995635000057
```

# Как трешолд и количество эмбэддингов в интервалах влияет на результирующую метрику?

`scripts/evaluate/process_query_hits.py`

Длинна одного интервала = 5 секунд^

Выводы: наибольшее значение итоговой метрики достгается при пересечении интервалов
примерно на половину или чуть больше половины.

Максимальное значение метрики = `0.776` с парамтерами:
```
query_interval_step         2.8
query_hits_intervals_step   14
threshold                   0.92
```

С экстримально маленькими интервалами (200, 400 мс) пересечения качество падает.
При пересеченнии стремящемся к длинне самого интервала тоже качество падает.

```
============================================
query_interval_step         1.0
query_hits_intervals_step   5
threshold 0.8   final_iou 0.496         f1 0.818        final_metric_value 0.617
threshold 0.82  final_iou 0.496         f1 0.818        final_metric_value 0.617
threshold 0.84  final_iou 0.496         f1 0.818        final_metric_value 0.618
threshold 0.86  final_iou 0.504         f1 0.821        final_metric_value 0.624
threshold 0.88  final_iou 0.515         f1 0.818        final_metric_value 0.632
threshold 0.9   final_iou 0.534         f1 0.828        final_metric_value 0.649
threshold 0.92  final_iou 0.561         f1 0.84         final_metric_value 0.673
threshold 0.94  final_iou 0.599         f1 0.852        final_metric_value 0.704  <-- max
threshold 0.96  final_iou 0.517         f1 0.824        final_metric_value 0.635
threshold 0.98  final_iou 0.245         f1 0.662        final_metric_value 0.358


============================================
query_interval_step         2.0
query_hits_intervals_step   10
threshold 0.8   final_iou 0.605         f1 0.889        final_metric_value 0.72
threshold 0.82  final_iou 0.61          f1 0.893        final_metric_value 0.725
threshold 0.84  final_iou 0.618         f1 0.896        final_metric_value 0.731  <-- max
threshold 0.86  final_iou 0.6           f1 0.884        final_metric_value 0.714
threshold 0.88  final_iou 0.594         f1 0.872        final_metric_value 0.706
threshold 0.9   final_iou 0.609         f1 0.874        final_metric_value 0.718
threshold 0.92  final_iou 0.62          f1 0.877        final_metric_value 0.726
threshold 0.94  final_iou 0.623         f1 0.88         final_metric_value 0.73
threshold 0.96  final_iou 0.486         f1 0.839        final_metric_value 0.616
threshold 0.98  final_iou 0.205         f1 0.629        final_metric_value 0.309


============================================
query_interval_step         2.4
query_hits_intervals_step   12
threshold 0.8   final_iou 0.641         f1 0.897        final_metric_value 0.748
threshold 0.82  final_iou 0.642         f1 0.897        final_metric_value 0.748
threshold 0.84  final_iou 0.641         f1 0.897        final_metric_value 0.748
threshold 0.86  final_iou 0.634         f1 0.898        final_metric_value 0.743
threshold 0.88  final_iou 0.636         f1 0.892        final_metric_value 0.742
threshold 0.9   final_iou 0.634         f1 0.883        final_metric_value 0.738
threshold 0.92  final_iou 0.656         f1 0.899        final_metric_value 0.758
threshold 0.94  final_iou 0.656         f1 0.902        final_metric_value 0.759
threshold 0.96  final_iou 0.586         f1 0.879        final_metric_value 0.703
threshold 0.98  final_iou 0.157         f1 0.64         final_metric_value 0.252


============================================
query_interval_step         2.6
query_hits_intervals_step   13
threshold 0.8   final_iou 0.578         f1 0.874        final_metric_value 0.696
threshold 0.82  final_iou 0.574         f1 0.871        final_metric_value 0.692
threshold 0.84  final_iou 0.584         f1 0.881        final_metric_value 0.703
threshold 0.86  final_iou 0.594         f1 0.888        final_metric_value 0.712
threshold 0.88  final_iou 0.6           f1 0.888        final_metric_value 0.716
threshold 0.9   final_iou 0.65          f1 0.9          final_metric_value 0.755
threshold 0.92  final_iou 0.639         f1 0.882        final_metric_value 0.741
threshold 0.94  final_iou 0.661         f1 0.903        final_metric_value 0.763
threshold 0.96  final_iou 0.568         f1 0.872        final_metric_value 0.688
threshold 0.98  final_iou 0.155         f1 0.648        final_metric_value 0.25


============================================
query_interval_step         2.8
query_hits_intervals_step   14
threshold 0.8   final_iou 0.658         f1 0.912        final_metric_value 0.764
threshold 0.82  final_iou 0.653         f1 0.908        final_metric_value 0.76
threshold 0.84  final_iou 0.653         f1 0.908        final_metric_value 0.76
threshold 0.86  final_iou 0.648         f1 0.905        final_metric_value 0.755
threshold 0.88  final_iou 0.647         f1 0.902        final_metric_value 0.753
threshold 0.9   final_iou 0.675         f1 0.908        final_metric_value 0.775
threshold 0.92  final_iou 0.675         f1 0.912        final_metric_value 0.776 <-- max
threshold 0.94  final_iou 0.661         f1 0.908        final_metric_value 0.765
threshold 0.96  final_iou 0.552         f1 0.877        final_metric_value 0.677
threshold 0.98  final_iou 0.143         f1 0.619        final_metric_value 0.232


============================================
query_interval_step         3.6
query_hits_intervals_step   18
threshold 0.8   final_iou 0.653         f1 0.908        final_metric_value 0.759
threshold 0.82  final_iou 0.653         f1 0.908        final_metric_value 0.76
threshold 0.84  final_iou 0.653         f1 0.908        final_metric_value 0.76
threshold 0.86  final_iou 0.631         f1 0.893        final_metric_value 0.74
threshold 0.88  final_iou 0.644         f1 0.902        final_metric_value 0.751
threshold 0.9   final_iou 0.649         f1 0.902        final_metric_value 0.755
threshold 0.92  final_iou 0.67          f1 0.917        final_metric_value 0.775  <-- max
threshold 0.94  final_iou 0.644         f1 0.905        final_metric_value 0.752
threshold 0.96  final_iou 0.418         f1 0.789        final_metric_value 0.546
threshold 0.98  final_iou 0.085         f1 0.575        final_metric_value 0.148


============================================
query_interval_step         4.0
query_hits_intervals_step   20
threshold 0.8   final_iou 0.63          f1 0.899        final_metric_value 0.741
threshold 0.82  final_iou 0.63          f1 0.899        final_metric_value 0.741  <-- max
threshold 0.84  final_iou 0.623         f1 0.893        final_metric_value 0.734
threshold 0.86  final_iou 0.616         f1 0.887        final_metric_value 0.727
threshold 0.88  final_iou 0.619         f1 0.884        final_metric_value 0.728
threshold 0.9   final_iou 0.63          f1 0.89         final_metric_value 0.738
threshold 0.92  final_iou 0.611         f1 0.884        final_metric_value 0.722
threshold 0.94  final_iou 0.486         f1 0.841        final_metric_value 0.616
threshold 0.96  final_iou 0.336         f1 0.766        final_metric_value 0.467
threshold 0.98  final_iou 0.117         f1 0.491        final_metric_value 0.189


============================================
query_interval_step         5.0
query_hits_intervals_step   25
threshold 0.8   final_iou 0.578         f1 0.864        final_metric_value 0.693
threshold 0.82  final_iou 0.573         f1 0.864        final_metric_value 0.689
threshold 0.84  final_iou 0.579         f1 0.867        final_metric_value 0.694
threshold 0.86  final_iou 0.578         f1 0.867        final_metric_value 0.694  <-- max
threshold 0.88  final_iou 0.568         f1 0.859        final_metric_value 0.684
threshold 0.9   final_iou 0.56          f1 0.855        final_metric_value 0.677
threshold 0.92  final_iou 0.469         f1 0.816        final_metric_value 0.595
threshold 0.94  final_iou 0.391         f1 0.794        final_metric_value 0.524
threshold 0.96  final_iou 0.233         f1 0.674        final_metric_value 0.346
threshold 0.98  final_iou 0.079         f1 0.471        final_metric_value 0.135
```

TODO Длинна интервала 10 секунд 

Увеличение длинны интервала немного удлинняет скорость генерации фингерпринтов.
В 2 раза.

Но возможно, увеличивает качество?

И в теории, можно оптимизировать сверточную часть сети для матчинга аудио.

