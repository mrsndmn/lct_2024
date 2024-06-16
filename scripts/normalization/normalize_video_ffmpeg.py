import argparse
import datetime
import time

from tqdm import tqdm
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


def sum_duration(directory):
    result = 0
    videos = os.listdir(directory)
    for video in videos:
        result += get_video_duration(directory + video)

    return result


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

def normalize_video_for_matching(
        input_video_path,
        output_video_path,
        target_fps=1,
        target_width=224,
        target_height=224,
        ):
    stream = ffmpeg.input(input_video_path). \
        filter('fps', fps=target_fps, round='up'). \
        filter('scale', w=target_width, h=target_height). \
        filter('hue', s=0)

    stream = ffmpeg.output(stream, output_video_path)
    ffmpeg.run(stream)

    trim_black_frames(output_video_path)

    return output_video_path


@mark_the_time
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", help="Target fps for normalization", type=int,
                        default=1)
    parser.add_argument("--width", help="Target width for normalization", type=int,
                        default=640)
    parser.add_argument("--height", help="Target height for normalization", type=int,
                        default=310)
    parser.add_argument("--duration", help="Min duration for applying normalization in seconds", type=int,
                        default=60)

    parser.add_argument("--source", help="Path to dir with source videos", type=str,
                        default="ex_videos/videos/")
    parser.add_argument("--target", help="Path to dir where you want to place result", type=str,
                        default="ex_videos/normalisation_result/")

    args = parser.parse_args()

    target_fps = args.fps
    target_width = args.width
    target_height = args.height
    min_duration = args.duration
    source_path = args.source
    target_path = args.target

    if source_path[-1] != "/": source_path += "/"
    if target_path[-1] != "/": target_path += "/"

    setupDir(target_path)

    videos = os.listdir(source_path)
    for video in tqdm(videos):
        video_path = source_path + video
        duration = get_video_duration(video_path)
        if duration < min_duration:
            print(f"{video} skippped because of too short duration: {duration}s")
            continue
        
        input_video_path = os.path.join(source_path, video)
        output_video_path = os.path.join(target_path, video)
        
        normalize_video_for_matching(
            input_video_path,
            output_video_path,
            target_fps=target_fps,
            target_width=target_width,
            target_height=target_height,
        )

if __name__ == '__main__':
    main()

# 1.37 hours - 5 videos
# 58 hours - all index

# ref 42.3
