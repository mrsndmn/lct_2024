import pickle
import os
from tempfile import TemporaryFile
from copy import deepcopy

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

def get_metrics(target: pd.DataFrame, submit: pd.DataFrame, debug=False):

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
    if debug:
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
            if debug:
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

    if debug:
        print("fp, fn, tp", fp, fn, tp)

    for ids, count in submit_dict.items():
        if ids not in target_dict:
            fp += count # модель нашла то, чего не было в оригинальной таблице
            # print("false_positive_ids", ids)

    if debug:
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

    # print("sorted_ious", sorted(ious, reverse=True))
    ious_no_zeros = [ iou for iou in ious if iou > 0.05]
    if debug:
        print("ious_no_zeros", sum(ious_no_zeros) / len(ious_no_zeros))

    f1_value = f1(tp, fp, fn)
    if debug:
        print(f'F1 = {f1_value}')

    final_iou = sum(ious) / (len(ious) + fp) # чтобы учесть количество лишних в IOU добавим в знаменатель их количество (так как их IOU = 0)
    if debug:
        print(f'IOU = {final_iou}')

    final_metric_value = final_metric(tp, fp, fn, final_iou)
    if debug:
        print(f'Metric = {final_metric_value}')

    return {
        "final_iou": final_iou,
        "f1": f1_value,
        "final_metric": final_metric_value,
    }

if __name__ == '__main__':


    # query_interval_step = 1.0
    # query_hits_dir = 'data/rutube/embeddings/electric-yogurt-97/search_val_embeddings/'

    # query_interval_step = 0.4
    # query_hits_dir = 'data/rutube/embeddings/electric-yogurt-97/search_val_embeddings_query_step_400ms/'

    # query_interval_step = 0.2
    # query_hits_dir = 'data/rutube/embeddings/electric-yogurt-97/search_val_embeddings_query_step_200ms/'

    query_hits_dir = 'data/rutube/embeddings/electric-yogurt-97/search_val_embeddings_query_step_200ms/'
    query_hits_files = os.listdir(query_hits_dir)

    # query_interval_step = 2.4
    # query_hits_intervals_step = 12
    query_intervals_by_file_name = dict()
    for query_hits_file in query_hits_files:
        query_hits_full_file_path = os.path.join(query_hits_dir, query_hits_file)

        with open(query_hits_full_file_path, 'rb') as f:
            query_hits_intervals = pickle.load(f)

        query_intervals_by_file_name[query_hits_file] = query_hits_intervals

    print("query_intervals_by_file_name", len(query_intervals_by_file_name))

    # for query_hits_intervals_step in [ 5, 10, 12, 13, 14, 18, 20, 25 ]:
    for query_hits_intervals_step in [ 12, 13, 14 ]:
        query_interval_step = round(query_hits_intervals_step * 0.2, 3)
        assert abs(0.2 * query_hits_intervals_step - query_interval_step) < 0.001

        print("\n\n============================================")
        print("query_interval_step        ", query_interval_step)
        print("query_hits_intervals_step  ", query_hits_intervals_step)

        for threshold in range(86, 100, 2):
            threshold = threshold / 100

            matched_intervals_for_queries = []
            for query_hits_file in query_hits_files:
                file_id = query_hits_file.split('.')[0]

                query_hits_intervals = query_intervals_by_file_name[query_hits_file]
                query_hits_intervals = query_hits_intervals[::query_hits_intervals_step]
                # print("len(query_hits_intervals)", len(query_hits_intervals))

                intervals_config = IntervalsConfig(
                    threshold=threshold,
                    query_interval_step=query_interval_step,
                    index_interval_step=1.0,
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

            got_metrics = get_metrics(target_df, created_df, debug=False)

            print("threshold", round(threshold, 2), "\tfinal_iou", round(got_metrics['final_iou'], 3), "\tf1", round(got_metrics['f1'], 3),  "\tfinal_metric_value", round(got_metrics['final_metric'], 3))

    raise Exception

