
# Техническая документация


# Глоссарий
* **Эмбэддинг** -- это вектор, который характеризует отрывок видео/аудио. Его генерит нейронка.
* **Фингерпринт** -- отпечаток аудио/видео, представляет из себя упорядоченную последовательность эмбэддингов

# Общая архитектура решения



# Процесс предобработки видео/аудио


# Процесс получения слепка


# Процесс загрузки/индексации набора видео в базу
![Процесс загрузки](img/fill_proc.png)
Отдельно обрабатываются аудио и кадры. Для каждого видео выполняется нормализация, формирование фингерпирнтов(эмбедингов) и их сохранение а БД.

# Процесс поиска видео по базе


# Основные взаимодействия с интерфейсом системы.
![UI](img/ui.png)
Для выбранного файла будет выведен список из N видео в которых найдены заимствования. 
Для каждого найденного видео можно переключаться между интервалами.

# Масштабируемость 

Сервис проверки заимстований (AVM) отвечает за нормализацию видео и извлечение эмбединогов, а также за бизнес логику по оределению заимстований. Этот сервис - stateless соответственно может легко масштабироваться как вертикально так и горизонтально без ограничений.  
Qdrant используется для хранения эмбедингов и поиска KNN. Может масштабироваться вертикально и горизонтально (шардирование). Возможности горизонтального масштабирования ограничены. 

-------


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

# Длинна интервала 10 секунд 

При увеличении длинны интервала получаем плохие значения метрик.
Возможно, из-за того, что моделька обучалась именно на 5-секундных интервалах.

Увеличение длинны интервала немного удлинняет скорость генерации фингерпринтов.
В 2 раза.

Но возможно, увеличивает качество?

И в теории, можно оптимизировать сверточную часть сети для матчинга аудио.

Время поиска похожих кандидатов - около часа для валидационного датасета на 8 ядрах.


```
query_interval_step         1
query_hits_intervals_step   1
threshold 0.86  final_iou 0.053         f1 0.128        final_metric_value 0.075
threshold 0.88  final_iou 0.088         f1 0.202        final_metric_value 0.122
threshold 0.9   final_iou 0.17           f1 0.35         final_metric_value 0.229
threshold 0.92  final_iou 0.234         f1 0.441        final_metric_value 0.306
threshold 0.94  final_iou 0.268         f1 0.498        final_metric_value 0.348
threshold 0.96  final_iou 0.304         f1 0.546        final_metric_value 0.391
threshold 0.98  final_iou 0.256         f1 0.524        final_metric_value 0.344


============================================
query_interval_step         2
query_hits_intervals_step   2
threshold 0.86  final_iou 0.071         f1 0.162        final_metric_value 0.099
threshold 0.88  final_iou 0.117         f1 0.253        final_metric_value 0.16
threshold 0.9   final_iou 0.211         f1 0.412        final_metric_value 0.279
threshold 0.92  final_iou 0.273         f1 0.498        final_metric_value 0.353
threshold 0.94  final_iou 0.279         f1 0.511        final_metric_value 0.361
threshold 0.96  final_iou 0.297         f1 0.544        final_metric_value 0.384
threshold 0.98  final_iou 0.209         f1 0.493        final_metric_value 0.293


============================================
query_interval_step         3
query_hits_intervals_step   3
threshold 0.86  final_iou 0.078         f1 0.189        final_metric_value 0.111
threshold 0.88  final_iou 0.126         f1 0.289        final_metric_value 0.176
threshold 0.9   final_iou 0.219         f1 0.449        final_metric_value 0.294
threshold 0.92  final_iou 0.26          f1 0.509        final_metric_value 0.344
threshold 0.94  final_iou 0.266         f1 0.527        final_metric_value 0.353
threshold 0.96  final_iou 0.277         f1 0.551        final_metric_value 0.368
threshold 0.98  final_iou 0.17          f1 0.491        final_metric_value 0.253


============================================
query_interval_step         4
query_hits_intervals_step   4
threshold 0.86  final_iou 0.09          f1 0.211        final_metric_value 0.126
threshold 0.88  final_iou 0.156         f1 0.341        final_metric_value 0.214
threshold 0.9   final_iou 0.232         f1 0.466        final_metric_value 0.31
threshold 0.92  final_iou 0.269         f1 0.516        final_metric_value 0.353
threshold 0.94  final_iou 0.268         f1 0.523        final_metric_value 0.355
threshold 0.96  final_iou 0.269         f1 0.55         final_metric_value 0.361
threshold 0.98  final_iou 0.152         f1 0.486        final_metric_value 0.231


============================================
query_interval_step         5
query_hits_intervals_step   5
threshold 0.86  final_iou 0.098         f1 0.234        final_metric_value 0.138
threshold 0.88  final_iou 0.16          f1 0.363        final_metric_value 0.222
threshold 0.9   final_iou 0.235         f1 0.476        final_metric_value 0.314
threshold 0.92  final_iou 0.252         f1 0.506        final_metric_value 0.337
threshold 0.94  final_iou 0.269         f1 0.536        final_metric_value 0.358
threshold 0.96  final_iou 0.269         f1 0.557        final_metric_value 0.363
threshold 0.98  final_iou 0.11          f1 0.431        final_metric_value 0.175


============================================
query_interval_step         6
query_hits_intervals_step   6
threshold 0.86  final_iou 0.108         f1 0.256        final_metric_value 0.152
threshold 0.88  final_iou 0.172         f1 0.387        final_metric_value 0.238
threshold 0.9   final_iou 0.259         f1 0.513        final_metric_value 0.344
threshold 0.92  final_iou 0.27          f1 0.534        final_metric_value 0.359
threshold 0.94  final_iou 0.267         f1 0.539        final_metric_value 0.357
threshold 0.96  final_iou 0.258         f1 0.554        final_metric_value 0.352
threshold 0.98  final_iou 0.122         f1 0.457        final_metric_value 0.192


============================================
query_interval_step         7
query_hits_intervals_step   7
threshold 0.86  final_iou 0.133         f1 0.306        final_metric_value 0.185
threshold 0.88  final_iou 0.208         f1 0.442        final_metric_value 0.283
threshold 0.9   final_iou 0.242         f1 0.498        final_metric_value 0.326
threshold 0.92  final_iou 0.26          f1 0.529        final_metric_value 0.349
threshold 0.94  final_iou 0.267         f1 0.549        final_metric_value 0.359
threshold 0.96  final_iou 0.242         f1 0.535        final_metric_value 0.333
threshold 0.98  final_iou 0.125         f1 0.444        final_metric_value 0.195


============================================
query_interval_step         8
query_hits_intervals_step   8
threshold 0.86  final_iou 0.117         f1 0.289        final_metric_value 0.167
threshold 0.88  final_iou 0.171         f1 0.392        final_metric_value 0.238
threshold 0.9   final_iou 0.206         f1 0.451        final_metric_value 0.283
threshold 0.92  final_iou 0.229         f1 0.492        final_metric_value 0.312
threshold 0.94  final_iou 0.235         f1 0.523        final_metric_value 0.324
threshold 0.96  final_iou 0.189         f1 0.498        final_metric_value 0.274
threshold 0.98  final_iou 0.058         f1 0.337        final_metric_value 0.099


============================================
query_interval_step         9
query_hits_intervals_step   9
threshold 0.86  final_iou 0.14          f1 0.335        final_metric_value 0.198
threshold 0.88  final_iou 0.197         f1 0.443        final_metric_value 0.273
threshold 0.9   final_iou 0.23          f1 0.498        final_metric_value 0.315
threshold 0.92  final_iou 0.236         f1 0.518        final_metric_value 0.324
threshold 0.94  final_iou 0.226         f1 0.523        final_metric_value 0.316
threshold 0.96  final_iou 0.179         f1 0.511        final_metric_value 0.265
threshold 0.98  final_iou 0.065         f1 0.386        final_metric_value 0.111


============================================
query_interval_step         10
query_hits_intervals_step   10
threshold 0.86  final_iou 0.156         f1 0.373        final_metric_value 0.22
threshold 0.88  final_iou 0.213         f1 0.478        final_metric_value 0.295
threshold 0.9   final_iou 0.235         f1 0.506        final_metric_value 0.321
threshold 0.92  final_iou 0.24          f1 0.52         final_metric_value 0.328
threshold 0.94  final_iou 0.247         f1 0.541        final_metric_value 0.339
threshold 0.96  final_iou 0.193         f1 0.525        final_metric_value 0.282
threshold 0.98  final_iou 0.076         f1 0.391        final_metric_value 0.127
```
