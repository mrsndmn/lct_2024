import os
import sys
from tqdm.auto import tqdm

# Для тестовых данных сприпт отработал за 2 минуты
# ~/miniconda3/envs/lct/bin/python scripts/data/extract_audio_from_video.py /Users/d.tarasov/Downloads/14.RUTUBE/compressed_test /Users/d.tarasov/Downloads/14.RUTUBE/compressed_test_audios

# Для индекса сприпт отработал за 4 минуты
# ~/miniconda3/envs/lct/bin/python scripts/data/extract_audio_from_video.py /Users/d.tarasov/Downloads/14.RUTUBE/compressed_index /Users/d.tarasov/Downloads/14.RUTUBE/compressed_index_audios

# Для данных из валидационного датасета тоже 4 минуты
# ~/miniconda3/envs/lct/bin/python scripts/data/extract_audio_from_video.py /Users/d.tarasov/Downloads/14.RUTUBE/compressed_val /Users/d.tarasov/Downloads/14.RUTUBE/compressed_val_audios

# $ du -hs /Users/d.tarasov/Downloads/14.RUTUBE/*_audios
# 6,3G    /Users/d.tarasov/Downloads/14.RUTUBE/compressed_index_audios
# 5,6G    /Users/d.tarasov/Downloads/14.RUTUBE/compressed_val_audios

if __name__ == '__main__':

    video_prefix = sys.argv[1]
    mp4_files = os.listdir(video_prefix)
    audio_prefix = sys.argv[2]

    print("len mp4_files", len(mp4_files))
    print("audio_prefix", audio_prefix)

    for mp4_file in tqdm(mp4_files):
        if not mp4_file.endswith(".mp4"):
            print("not mp4 file:", mp4_file)
            continue

        print("process", mp4_file)
        output_file_name = mp4_file.replace(".mp4", ".wav")
        output_file_name = audio_prefix + "/" + output_file_name

        video_full_path = video_prefix + '/' + mp4_file
        exit_code = os.system(f"ffmpeg -i {video_full_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {output_file_name}")
        assert exit_code == 0, f"{exit_code} != 0; mp4_file={video_full_path}, output_file_name={output_file_name}"

