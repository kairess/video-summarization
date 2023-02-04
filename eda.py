import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

CB91_Blue = "#2CBDFE"
CB91_Green = "#47DBCD"
CB91_Pink = "#F3A0F2"
CB91_Purple = "#9D2EC5"
CB91_Violet = "#661D98"
CB91_Amber = "#F5B14C"

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber, CB91_Purple, CB91_Violet]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=color_list)
plt.rcParams["font.family"] = "NanumBarunGothic"


class Annotation:
    def __init__(self, annotation_file):
        self.file_name = annotation_file
        with open(annotation_file, "r") as rf:
            data = json.load(rf)
        self.metadata = data["metadata"]
        self.timelines = data["timelines"]


def normalize(arr):
    arr_min, arr_max = arr.min(), arr.max()
    if arr_min == arr_max:
        return arr / arr.shape[0]
    return (arr - arr_min) / (arr_max - arr_min)


def percentize(arr):
    sum = arr.sum()
    return arr / sum * 100


def plot_keyshot_distributions(annotations):

    keyshot_heatmap = np.zeros(100)
    keyshot_start = np.zeros(100)
    keyshot_middle = np.zeros(100)
    keyshot_end = np.zeros(100)

    for annotation in annotations:

        length = annotation.metadata["length"]

        for keyshot in annotation.timelines:

            start = keyshot["start"]
            end = keyshot["end"]

            if start > end:
                continue

            key_segments = np.arange(start, end + 1) / length * 100
            key_segments = np.trunc(key_segments)

            keyshot_heatmap[key_segments.astype(int)] += 1
            keyshot_start[np.trunc(start / length * 100).astype(int)] += 1
            keyshot_middle[np.trunc((start + end) / 2 / length * 100).astype(int)] += 1
            keyshot_end[np.trunc(end / length * 100).astype(int)] += 1

    percentized_keyshot_heatmap = percentize(keyshot_heatmap)
    keyshot_start = percentize(keyshot_start)
    keyshot_middle = percentize(keyshot_middle)
    keyshot_end = percentize(keyshot_end)

    fig, ax = plt.subplots(5, sharex=True, figsize=(10, 20))
    fig.suptitle(f"주요 장면 분포", style="oblique", y=0.99)
    plt.xlabel("비디오 진행 (%)")
    ax[0].bar(np.linspace(0, 100, num=100), keyshot_heatmap)
    ax[0].set_ylabel("주요장면")
    ax[1].bar(np.linspace(0, 100, num=100), percentized_keyshot_heatmap)
    ax[1].set_ylabel("주요장면 (%)")
    ax[2].bar(np.linspace(0, 100, num=100), keyshot_middle)
    ax[2].set_ylabel("주요장면 중간 (%)")
    ax[3].bar(np.linspace(0, 100, num=100), keyshot_start)
    ax[3].set_ylabel("주요장면 시작 (%)")
    ax[4].bar(np.linspace(0, 100, num=100), keyshot_end)
    ax[4].set_ylabel("주요장면 종료 (%)")

    for i in range(5):
        ax[i].xaxis.grid(True, which="major")
        ax[i].yaxis.grid(True, which="major")

    fig.tight_layout()
    fig.savefig("keyshot_distributions.png")


def plot_keyshot_durations(annotations):

    durations = []

    for annotation in annotations:

        length = annotation.metadata["length"]

        for keyshot in annotation.timelines:

            start = keyshot["start"]
            end = keyshot["end"]

            if start > end:
                continue

            duration = (end - start) * 100 / length
            durations.append(duration)

    fig, ax = plt.subplots()
    plt.title(f"주요 장면 비중 분포", style="oblique", y=0.99)
    # plt.xlabel("비디오 진행 (%)")
    ax.hist(durations, bins=30)
    ax.set_ylabel("주요장면 비중 (%)")

    fig.tight_layout()
    fig.savefig("keyshot_durations.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-dir", help="Directory with annotation and video files"
    )
    args = parser.parse_args()

    annotations = [Annotation(f) for f in Path(args.data_dir).glob("**/*.json")]

    plot_keyshot_distributions(annotations)
    plot_keyshot_durations(annotations)
