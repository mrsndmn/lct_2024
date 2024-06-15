import pickle
import os
from tempfile import TemporaryFile

import pandas as pd
from scripts.evaluate.borrow_intervals import IntervalsConfig
from avm.matcher import Segment, MatchedSegmentsPair, get_matched_segments


def parse_segment(segment):
    start, end = map(int, segment.split("-"))
    return start, end


def iou(segment_q, segment_t):
    start_q, stop_q = parse_segment(segment_q)
    start_t, stop_t = parse_segment(segment_t)

    intersection_start = max(start_q, start_t)
    intersection_end = min(stop_q, stop_t)

    intersection_length = max(0, intersection_end - intersection_start)
    union_length = (stop_q - start_q) + (stop_t - start_t) - intersection_length

    iou = intersection_length / union_length if union_length > 0 else 0
    return iou


def f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return 2 * (precision * recall) / (precision + recall + 1e-6)

def final_metric(tp, fp, fn, final_iou):
    f = f1(tp, fp, fn)

    return 2 * (final_iou * f) / (final_iou + f + 1e-6)

def get_metrics(target: pd.DataFrame, submit: pd.DataFrame):

    target['ID-piracy'] = target['ID_piracy'].map(lambda x: x.removesuffix(".mp4"))
    target['SEG-piracy'] = target['segment']
    target['ID-license'] = target['ID_license'].map(lambda x: x.removesuffix(".mp4"))
    target['SEG-license'] = target['segment.1']

    target['SEG-piracy-start'] = target['SEG-piracy'].map(lambda x: int(x.split("-")[0]))
    target['SEG-piracy-end'] = target['SEG-piracy'].map(lambda x: int(x.split("-")[1]))
    target['SEG-license-end'] = target['SEG-license'].map(lambda x: int(x.split("-")[1]))
    target['SEG-license-start'] = target['SEG-license'].map(lambda x: int(x.split("-")[0]))    

    created_df['SEG-piracy-start'] = created_df['SEG-piracy'].map(lambda x: int(x.split("-")[0]))
    created_df['SEG-piracy-end'] = created_df['SEG-piracy'].map(lambda x: int(x.split("-")[1]))
    created_df['SEG-license-end'] = created_df['SEG-license'].map(lambda x: int(x.split("-")[1]))
    created_df['SEG-license-start'] = created_df['SEG-license'].map(lambda x: int(x.split("-")[0]))
    #  (created_df['SEG-piracy-end'] - created_df['SEG-piracy-start']).describe()
    #  (created_df['SEG-license-end'] - created_df['SEG-license-start']).describe()

    target_dict = target.groupby(['ID-piracy', 'ID-license']).count().to_dict()['SEG-piracy']

    submit_dict = submit.groupby(['ID-piracy', 'ID-license']).count().to_dict()['SEG-piracy']
    print('len(target)\t', len(target))
    print('len(submit)\t', len(submit))
    print("len(orig_dict)\t", len(target_dict))
    print("len(target_dict)\t", len(submit_dict))

    """# Подсчет FP, TP, FN"""

    fn = 0
    fp = 0
    tp = 0

    for ids, count in target_dict.items():
        if ids not in submit_dict:
            fn += count # модель не нашла что то из оригинальной таблицы
            print("false negative:", ids)
            # breakpoint()
        elif submit_dict[ids] > count:
            fp += submit_dict[ids] - count     # модель нашла больше совпадений чем в оригинальной таблице
            tp += min(submit_dict[ids], count) # тогда для истинных совпадений совпадений берем наименьшее количество
        elif submit_dict[ids] < count:
            fn += count - submit_dict[ids]     # модель нашла меньше совпадений чем в оригинальной таблице
            tp += min(submit_dict[ids], count) # тогда для истинных совпадений совпадений берем наименьшее количество
        elif submit_dict[ids] == count:
            tp += count                        # если количество совпало, должны засчитать их как true positive

    print("fp, fn, tp", fp, fn, tp)

    for ids, count in submit_dict.items():
        if ids not in target_dict:
            fp += count # модель нашла то, чего не было в оригинальной таблице
            # print("false_positive_ids", ids)

    print("fp, fn, tp", fp, fn, tp)

    """# Подсчет IOU"""

    ious = []

    # Подсчет IOU для каждой отдельной строки из orig
    for i, row in target.iterrows():
        max_iou = 0
        merged = pd.merge(
            row.to_frame().T,
            submit,
            'left',
            left_on=['ID-piracy', 'ID-license'],
            right_on=['ID-piracy', 'ID-license']
        ).dropna()

        # Выбор наилучшего IOU по всем совпадениям из target
        if len(merged) > 0:
            for j, row1 in merged.iterrows():
                final_iou = iou(row1['SEG-piracy_x'], row1['SEG-piracy_y']) * iou(row1['SEG-license_x'], row1['SEG-license_y'])
                if final_iou > max_iou:
                    max_iou = final_iou

        ious.append(max_iou)

    print("sorted_ious", sorted(ious, reverse=True))
    ious_no_zeros = [ iou for iou in ious if iou > 0.05]
    print("ious_no_zeros", sum(ious_no_zeros) / len(ious_no_zeros))

    print(f'F1 = {f1(tp, fp, fn)}')

    final_iou = sum(ious) / (len(ious) + fp) # чтобы учесть количество лишних в IOU добавим в знаменатель их количество (так как их IOU = 0)
    print(f'IOU = {final_iou}')

    print(f'Metric = {final_metric(tp, fp, fn, final_iou)}')

if __name__ == '__main__':

    query_hits_dir = 'data/rutube/embeddings/electric-yogurt-97/search_val_embeddings/'
    # query_hits_dir = 'data/rutube/embeddings/electric-yogurt-97/search_val_embeddings_query_step_400ms/'
    query_hits_files = os.listdir(query_hits_dir)

    matched_intervals_for_queries = []
    for query_hits_file in query_hits_files:
        query_hits_full_file_path = os.path.join(query_hits_dir, query_hits_file)
        file_id = query_hits_file.split('.')[0]

        with open(query_hits_full_file_path, 'rb') as f:
            query_hits_intervals = pickle.load(f)

        # query_embeddings = query_embeddings[::2]
        # print(len(query_hits_intervals))

        intervals_config = IntervalsConfig(
            threshold=0.93,
            index_interval_step=1.0,
            query_interval_step=1.0,
            merge_segments_with_diff_seconds=10.0,
            interval_duration_in_seconds=5,
            segment_min_duration=20,
        )
        matched_intervals = get_matched_segments(intervals_config, file_id, query_hits_intervals)

        matched_intervals_for_queries.append(matched_intervals)

    validate_file_items = []
    for matched_intervals in matched_intervals_for_queries:
        for mi in matched_intervals:
            mi: MatchedSegmentsPair
            piracy_interval: Segment = mi.current_segment
            license_interval: Segment = mi.licensed_segment
            validate_file_item = {
                "ID-piracy": piracy_interval.file_id,
                "SEG-piracy": piracy_interval.format_string(),
                "ID-license": license_interval.file_id,
                "SEG-license": license_interval.format_string(),
            }
            validate_file_items.append(validate_file_item)

    created_df = pd.DataFrame(validate_file_items)

    target_df = pd.read_csv('data/rutube/piracy_val.csv')

    get_metrics(target_df, created_df)

    raise Exception

