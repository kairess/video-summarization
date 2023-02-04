import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm
from transformers import ViTFeatureExtractor


def extract_video_features(extractor, video_file, sample_every):

    vc = cv2.VideoCapture(str(video_file))
    fps = int(vc.get(cv2.CAP_PROP_FPS))
    frames = []
    last_collected = -1
    while vc.isOpened():

        success, frame = vc.read()
        if not success:
            break

        timestmap = vc.get(cv2.CAP_PROP_POS_MSEC)
        second = timestmap // 1000
        if second != last_collected:
            last_collected = second
            frames.append(frame)

    features = extractor(images=frames, return_tensors="pt")
    return features["pixel_values"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-directory", help="Directory with data")
    parser.add_argument("-o", "--out", help="Output h5 file")
    parser.add_argument("-s", "--sample-every", help="Sampling rate", default=-1)
    args = parser.parse_args()

    video_files = tqdm(list(Path(args.data_directory).glob("**/*.mp4")))
    extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224", size=224
    )

    with h5py.File(args.out, "w") as wf:

        for video_file in video_files:
            name = video_file.stem
            try:
                features = extract_video_features(
                    extractor, video_file, sample_every=args.sample_every
                )
                wf.create_dataset(name, data=features)
            except Exception as e:
                print(e)
