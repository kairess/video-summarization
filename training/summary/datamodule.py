import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset


class SummaryDataset(Dataset):
    def __init__(self, video_names, directory, video_features_file, max_seq_len=250):
        self.directory = directory
        self.video_names = video_names
        self.video_features_file = video_features_file
        self.max_seq_len = max_seq_len

        # Because we can't use DDP with IterableDataset,
        # data must be pre-chunked to combat OOM.
        self.label_files = self.prefetch_label_files()
        self.data_size, self.index_to_chunk, self.labels = self.prefetch_and_index()

    def prefetch_label_files(self):

        name_set = set(self.video_names)

        label_files = defaultdict(list)

        for label_file in Path(self.directory).glob(f"**/*.json"):

            file_name = label_file.stem

            # 예시: [KBS]kim370_대법원 업무 과부하…상고 법원이 대안_18567498.json
            # annotator id 제거하면 비디오 이름 추출.
            # 파일 이름 reverse ([::-1]) 후 "_" 찾음.
            annotator_id_index = len(file_name) - file_name[::-1].find("_") - 1
            video_name = file_name[:annotator_id_index]

            if video_name in name_set:
                label_files[video_name].append(label_file)

        return label_files

    def prefetch_and_index(self):
        index = 0
        index_to_chunk = {}
        all_labels = {}

        for video_name in self.video_names:

            if video_name == "news_footage_1710":
                continue

            labels = self.extract_label(video_name)
            all_labels[video_name] = labels

            chunk_count = math.ceil(len(labels[0]) / self.max_seq_len)
            for chunk_index in range(0, chunk_count):
                index_to_chunk[index + chunk_index] = (video_name, chunk_index)

            index += chunk_count

        return index, index_to_chunk, all_labels

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):

        video_name, chunk_index = self.index_to_chunk[index]
        start = chunk_index * self.max_seq_len
        end = start + self.max_seq_len

        with h5py.File(self.video_features_file, "r") as rf:

            labels = self.labels[video_name]
            video_features = rf[video_name][()][: len(labels[0])][start:end]
            labels = labels[:, start:end]
            return video_name, video_features, labels

    def extract_label(self, video_name):

        label_files = self.label_files[video_name]
        labels = []

        for label_file in label_files:

            with open(label_file, "r") as rf:
                data = json.load(rf)

            metadata = data["metadata"]
            video_length = math.ceil(metadata["length"])
            annotator_label = np.zeros((video_length,))

            for timeline in data["timelines"]:
                for time_index in range(timeline["start"], timeline["end"] + 1):
                    # annotator_label[time_index] += 1
                    if time_index < video_length:
                        annotator_label[time_index] = 1

            labels.append(annotator_label)

        labels = np.array(labels)
        return labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-directory", help="Data directory")
    parser.add_argument("-v", "--video-features-file", help="Video features file")
    args = parser.parse_args()

    videos = ["유튜브_일상_13450", "news_footage_0771", "유튜브_반려동물및동물_3220"]
    sd = SummaryDataset(videos, args.data_directory, args.video_features_file)
    dl = DataLoader(sd, batch_size=1)
    for d in dl:
        print(d[0])
        pass
