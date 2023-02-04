import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


class Annotation:
    def __init__(self, annotation_file):
        self.file_name = annotation_file
        with open(annotation_file, "r") as rf:
            data = json.load(rf)
        self.metadata = data["metadata"]
        self.video_name = Path(self.metadata["file_name"]).stem
        self.annotator = self.metadata["annotator_id"]
        self.timelines = data["timelines"]
        self.has_error = False


class Timeline:
    def __init__(self, video_name, annotator, timeline):
        self.video_name = video_name
        self.annotator = annotator
        self.id = timeline["id"]
        self.start = timeline["start"]
        self.end = timeline["end"]
        self.attributes = timeline["attributes"]
        self.place = self.attributes["place"]
        self.action = self.attributes["action"]
        self.emotion = self.attributes["emotion"]
        self.relationship = self.attributes["relationship"]

    def overlaps(self, other):
        return (
            other.start <= self.start <= other.end
            or self.start <= other.start <= self.end
        )


def check_individual_annotation(annotation):

    name = annotation.file_name
    length = annotation.metadata["length"]
    class_code = annotation.metadata["class_code"]

    # Check if timelines ids overlap.
    if len(annotation.timelines) != len({t["id"] for t in annotation.timelines}):
        print(f"[ERROR] 주요 장면 id 겹침: {str(annotation.file_name)}.")

    # Check if class code is within range.
    if type(class_code) is not int or not 1 <= class_code <= 8:
        annotation.has_error = True
        print(f"[ERROR] class_code 범위 에러: {str(annotation.file_name)}.")

    total_annotation_duration = 0

    for timeline in annotation.timelines:

        id = timeline["id"]
        start = timeline["start"]
        end = timeline["end"]
        attributes = timeline["attributes"]
        place = attributes["place"]
        action = attributes["place"]
        emotion = attributes["emotion"]
        relationship = attributes["relationship"]

        # Check length, start, and end formats.
        if type(length) is not float:
            annotation.has_error = True
            print(f"[ERROR] 영상 길이 표기가 float이 아님: {name}")
        if type(start) is not int:
            annotation.has_error = True
            print(f"[ERROR] 주요 장면 시작 표기가 int가 아님: {name}")
        if type(end) is not int:
            annotation.has_error = True
            print(f"[ERROR] 주요 장면 종료 표기가 int가 아님: {name}")

        # Check if annotation start and end is within range.
        if start < 0:
            annotation.has_error = True
            print(f"[ERROR] 주요 장면 시작 범위 오류: {name}")
        if end >= length:
            annotation.has_error = True
            print(f"[ERROR] 주요 장면 종료 범위 오류: {name}")

        # Check if annotation start is smaller than end.
        if start > end:
            annotation.has_error = True
            print(f"[ERROR] 주요 장면 시작 > 주요 장면 종료: {name}")
        else:
            total_annotation_duration += end - start

        # Check if each tag is within range.
        if type(place) is not int or not 1 <= place <= 21:
            annotation.has_error = True
            print(f"[ERROR] place attribute 범위 에러: {name} (timeline id {id})")
        if type(action) is not int or not 1 <= action <= 65:
            annotation.has_error = True
            print(f"[ERROR] action attribute 범위 에러: {name} (timeline id {id})")
        if type(emotion) is not int or not 1 <= emotion <= 7:
            annotation.has_error = True
            print(f"[ERROR] emotion attribute 범위 에러: {name} (timeline id {id})")
        if type(relationship) is not int or not 1 <= relationship <= 12:
            annotation.has_error = True
            print(f"[ERROR] relationship attribute 범위 에러: {name} (timeline id {id})")

    # Check annotation count.
    tag_count = round(length // 60 / 2)
    if len(annotation.timelines) < tag_count:
        annotation.has_error = True
        print(f"[ERROR] 주요 장면 태그 개수 오류: {name}")

    # Check annotation percentage.
    annotation_percentage = total_annotation_duration * 100 / length
    if math.ceil(annotation_percentage) < 5:
        annotation.has_error = True
        print(f"[ERROR] 주요 장면 5% 미만: {name}")
    if int(annotation_percentage) > 25:
        annotation.has_error = True
        print(annotation_percentage, total_annotation_duration, length)
        print(f"[ERROR] 주요 장면 25% 초과: {name}")


def check_annotation_group(annotations):

    keys_to_compare = {
        "file_name",
        "type",
        "length",
        "quality",
        "date",
        "license",
        "class_code",
    }

    prev_metadata, annotator_ids = None, set()

    # Collect all timelines.
    timelines = []
    length = math.ceil(annotations[0].metadata["length"])
    timeline_marks = [set() for _ in range(length)]
    for annotation in annotations:
        for t in annotation.timelines:
            timeline = Timeline(annotation.video_name, annotation.annotator, t)
            timelines.append(timeline)
            for i in range(timeline.start, timeline.end + 1):
                if i < length:
                    timeline_marks[i].add(timeline)

    groups = []
    running_group = set()
    for i, timeline_set in enumerate(timeline_marks):
        if timeline_set:
            running_group = running_group.union(timeline_set)
        else:
            if running_group:
                groups.append(running_group)
            running_group = set()
    if running_group:
        groups.append(running_group)

    for group in groups:
        if len(group) < 2:
            continue
        group_list = list(group)
        invalid_timelines = set()
        for t1, t2 in zip(group_list, group_list[1:]):
            video_name = t1.video_name
            if (
                t1.place != t2.place
                or t1.action != t2.action
                or t1.emotion != t2.emotion
                or t1.relationship != t2.relationship
            ):
                invalid_timelines.add(t1)
                invalid_timelines.add(t2)

        if invalid_timelines:
            t_ids = [
                f"annotator id {t.annotator} timeline id {t.id}"
                for t in invalid_timelines
            ]
            error_string = f"{video_name} - {', '.join(t_ids)}"

            # print(f"[ERROR] 겹치는 annotation 태그 불일치: {error_string}.")
            # print(f"{error_string}")

    # Check annotations.
    for annotation in annotations:
        metadata = {
            key: value
            for key, value in annotation.metadata.items()
            if key in keys_to_compare
        }
        annotator_id = annotation.metadata["annotator_id"]

        # Check if annotator ids overlap.
        if annotator_id in annotator_ids:
            print(
                f"[ERROR] annotator_id 겹침: {[str(a.file_name) for a in annotations]}."
            )
        annotator_ids.add(annotator_id)

        # Check if metadata content is same (except for annotator id).
        if prev_metadata is not None and metadata != prev_metadata:
            print(f"[ERROR] metadata 불일치: {[(a.file_name) for a in annotations]}.")

        prev_metadata = metadata


#        timeline_groups = [set()] * metadata
# Check overlap.
#        for current in annotation.timelines:
#            for stored in timelines:
#                if current.overlaps(stored):
#                    timeline_group = timeline_groups["current"]
#                    timeline_group.timelines.add(current)
#                    timeline_groups[current] = timeline_group


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", help="Directory with json tag data")
    args = parser.parse_args()

    # Display all file extensions.
    extensions = {file.suffix for file in Path(args.data_dir).glob("**/*")}
    print(f"데이터 디렉토리 내의 파일 확장자: {extensions}")

    # List of all json files.
    annotation_files = list(Path(args.data_dir).glob("**/*.json"))

    # List of all video files.
    video_files = list(Path(args.data_dir).glob("**/*.mp4"))
    video_names = [p.stem for p in video_files]

    # Display file count.
    print(f"메타데이터 (json) 파일 개수: {len(annotation_files)}")
    print(f"비디오 (.mp4) 파일 개수: {len(video_files)}")

    # Check if all annotation file names follow a consistent format
    # regarding their corresponding video filenames.
    name_to_annotation = defaultdict(list)
    annotation_names = set()
    for file in tqdm(Path(args.data_dir).glob("**/*.json"), total=102001):
        annotation = Annotation(file)
        name = Path(annotation.metadata["file_name"]).stem
        name_to_annotation[name].append(annotation)
        annotation_names.add(name)

    # Display annotation count distribution.
    annotation_counts = defaultdict(int)
    for name, annotations in name_to_annotation.items():
        annotation_counts[f"{len(annotations)}"] += 1
    print(f"어노테이션 개수: {[f'{k}개: {v}' for k,v in annotation_counts.items()]}")

    # Check if every json file has a corresponding video file, and vice versa.
    annotation_set, video_set = set(annotation_names), set(video_names)
    only_in_annotation = annotation_set - video_set
    if only_in_annotation:
        print(f"[ERROR] 메타데이터만 존재하는 파일: {len(only_in_annotation)}개")
    only_in_video = video_set - annotation_set
    if only_in_video:
        print(f"[ERROR] 비디오만 존재하는 파일: {len(only_in_video)}개")

    # Check annotation validity.
    for name, annotations in name_to_annotation.items():
        for annotation in annotations:
            check_individual_annotation(annotation)
        check_annotation_group(annotations)
