import argparse

import cv2


def extract_video_features(extractor, video_file, sample_every=-1):

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
