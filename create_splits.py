import argparse
import json
import random
from pathlib import Path

import h5py


def extract_video_name(label_file):
    # 예시: [KBS]kim370_대법원 업무 과부하…상고 법원이 대안_18567498.json
    # annotator id 제거하면 비디오 이름 추출.
    # 파일 이름 reverse ([::-1]) 후 "_" 찾음.
    file_name = label_file.stem
    annotator_id_index = len(file_name) - file_name[::-1].find("_") - 1
    video_name = file_name[:annotator_id_index]
    return video_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-ld", "--labels-dir", help="Directory with labels")
    parser.add_argument("-f", "--video-features", help="h5 file with video features")
    parser.add_argument("-o", "--out-file", help="Output json file")
    args = parser.parse_args()

    label_files = list(Path(args.labels_dir).glob("**/*.json"))
    label_video_names = [extract_video_name(p) for p in label_files]

    with h5py.File(args.video_features) as rf:
        feature_video_names = {key for key in rf.keys()}

    video_names = set(
        [name for name in label_video_names if name in feature_video_names]
    )
    video_names = list(video_names)

    train_count = int(len(video_names) * 0.9)
    val_count = (len(video_names) - train_count) // 2
    test_count = len(video_names) - train_count - val_count

    assert train_count + val_count + test_count == len(video_names)

    print(
        f"Train: {train_count} samples, Validate: {val_count} samples, Test: {test_count} samples."
    )

    random.shuffle(video_names)
    train_split = video_names[:train_count]
    val_split = video_names[train_count : train_count + val_count]
    test_split = video_names[-test_count:]

    assert len(train_split) + len(val_split) + len(test_split) == len(video_names)

    data = {"train": train_split, "validate": val_split, "test": test_split}

    with open(args.out_file, "w") as wf:
        json.dump(data, wf, indent=4, ensure_ascii=False)
