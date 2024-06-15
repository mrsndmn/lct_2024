import datetime
import time

from DPF.configs import FilesDatasetConfig
from DPF.dataset_reader import DatasetReader
from DPF.transforms import VideoFFMPEGTransforms, Resizer, ResizerModes
from distutils.dir_util import copy_tree
import ffmpeg
import subprocess as sp
import re

import os
import cv2
import shutil


def trim_black_frames(video_path: str):
    in_video_file = video_path
    out_video_file = video_path

    # ffmpeg -hide_banner -i wnkaa.mp4 -vf cropdetect=skip=0 -t 1 -f null
    cropdetect_output = sp.run(
        ['ffmpeg', '-hide_banner', '-i', in_video_file, '-vf', 'cropdetect=skip=0', '-t', '1', '-f', 'null', 'pipe:'],
        stderr=sp.PIPE, universal_newlines=True).stdout
    if not cropdetect_output:
        return

    # Return: crop=480:304:72:4
    crop_str = re.search('crop=.*', cropdetect_output)
    if not crop_str:
        return

    crop_str.group(0)
    # ffmpeg -hide_banner -i wnkaa.mp4 -vf crop=480:304:72:4,setsar=1 cropped_wnkaa.mp4
    sp.run(['ffmpeg', '-hide_banner', '-i', in_video_file, '-vf', crop_str + ',setsar=1', out_video_file])

    print("trimmed frames for video", video_path)


def get_video_duration(video_path: str) -> float:
    # using only opencv-python package, fast but can be inaccurate
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps  # in seconds
    cap.release()
    return duration


def setupDir(directory: str):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    else:
        shutil.rmtree(directory)  # clear dir content
        os.mkdir(directory)


def mark_the_time(f):
    def wrapper():
        start = time.time()
        f()
        end = time.time()
        print(f"Execution time is {str(datetime.timedelta(seconds=end - start))}")

    return wrapper


@mark_the_time
def main():
    target_fps = 5
    target_width = 640
    target_height = 310
    min_duration = 60  # seconds

    source_path = "ex_videos/videos/"
    target_path = "ex_videos/normalisation_result/"

    setupDir(target_path)

    videos = os.listdir(source_path)
    for video in videos:
        video_path = source_path + video
        duration = get_video_duration(video_path)
        if duration < min_duration:
            print(f"{video} skippped because of too short duration: {duration}s")
            continue

        stream = ffmpeg.input(video_path).filter('fps', fps=target_fps, round='up').filter('scale',
                                                                                     w=target_width,
                                                                                     h=target_height)
        new_video_path = target_path + video
        stream = ffmpeg.output(stream, new_video_path)
        ffmpeg.run(stream)

        trim_black_frames(new_video_path)


if __name__ == '__main__':
    main()
