
# Скачиваем индексы для аудио и видео
wget https://storage.yandexcloud.net/lct2024-task-14/audio_index_embeddings.zip
wget https://storage.yandexcloud.net/lct2024-task-14/video-index.zip

# веса для аудио-модельки
wget https://storage.yandexcloud.net/lct2024-task-14/electric-yogurt-97.zip

# веса для визуальной модальности
wget https://storage.yandexcloud.net/lct2024-task-14/ruby-moon-17.zip

wget https://storage.yandexcloud.net/lct2024-task-14/compressed_normalized_index.zip


# Videos data
mkdir -p data/rutube/videos/
unzip compressed_normalized_index.zip -d data/rutube/videos/

# Video Model
mkdir -p data/models/image/efficient-net-b0/
unzip ruby-moon-17.zip -d data/models/image/efficient-net-b0/

# Audio model
mkdir -p data/models/UniSpeechSatForXVector_mini_finetuned
unzip electric-yogurt-97.zip -d data/models/UniSpeechSatForXVector_mini_finetuned

# Audio Embeddings
mkdir -p data/rutube/embeddings/electric-yogurt-97
unzip audio_index_embeddings.zip -d data/rutube/embeddings/electric-yogurt-97

# Video Embeddings
mkdir -p data/rutube/embeddings/video/
unzip video-index.zip -d data/rutube/embeddings/video/
