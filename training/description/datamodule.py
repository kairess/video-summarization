import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Annotation:
    def __init__(self, video_name, timestamp, attributes):
        self.video_name = video_name
        self.timestamp = timestamp
        self.place = int(attributes["place"]) - 1
        self.action = int(attributes["action"]) - 1
        self.emotion = int(attributes["emotion"]) - 1
        self.relationship = int(attributes["relationship"]) - 1


class DescriptionDataset(Dataset):
    def __init__(self, video_names, directory, video_features_file, max_seq_len=250):
        self.directory = directory
        self.video_names = video_names
        self.video_features_file = video_features_file
        self.max_seq_len = max_seq_len
        self.annotations = self.prefetch_annotations()

        # Because we can't use DDP with IterableDataset,
        # data must be pre-chunked to combat OOM.
        # self.data = self.prefetch_annotations()
        # self.data_size, self.index_to_chunk, self.labels = self.prefetch_and_index()

    def prefetch_annotations(self):
        name_set = set(self.video_names)
        data = {}
        index = 0
        for label_file in tqdm(
            Path(self.directory).glob(f"**/*.json"), desc="Prefetching data..."
        ):
            file_name = label_file.stem
            # 예시: [KBS]kim370_대법원 업무 과부하…상고 법원이 대안_18567498.json
            # annotator id 제거하면 비디오 이름 추출.
            # 파일 이름 reverse ([::-1]) 후 "_" 찾음.
            annotator_id_index = len(file_name) - file_name[::-1].find("_") - 1
            video_name = file_name[:annotator_id_index]
            if video_name in name_set:
                with open(label_file, "r") as rf:
                    json_data = json.load(rf)
                timelines = json_data["timelines"]
                for timeline in timelines:
                    start, end = timeline["start"], timeline["end"]
                    attributes = timeline["attributes"]
                    timestamp = (end + start) // 2
                    annotation = Annotation(video_name, timestamp, attributes)
                    data[index] = annotation
                    index += 1
        return data

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        annotation = self.annotations[index]
        try:
            with h5py.File(self.video_features_file, "r") as vf:
                video_feature = vf[annotation.video_name][()][annotation.timestamp]
            place = int(annotation.place)
            action = int(annotation.action)
            emotion = int(annotation.emotion)
            relationship = int(annotation.relationship)
        except Exception as e:
            # print(f"Corruption in video {annotation.video_name}")
            print(e)
            place = action = emotion = relationship = 0
            video_feature = torch.rand((3, 224, 224))
        return (
            annotation.video_name,
            video_feature,
            place,
            action,
            emotion,
            relationship,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-directory", help="Video files directory")
    parser.add_argument("-v", "--video-features-file", help="Video files directory")
    args = parser.parse_args()

    videos = [
        "[KBS]lee794_현대사의 한 축…파란만장 정치 역정",
        "[KBS]_이슈_현장__알파고_대결_이후_더_강해진_이세돌_HQ_20210826V98156",
        "[KBS]_이슈_현장__영정_사진_프로젝트__장수를_기원합니다__HQ_20210826V98120",
    ]
    dd = DescriptionDataset(videos, args.data_directory, args.video_features_file)
    dl = DataLoader(dd, batch_size=1)
    for _ in dl:
        pass
