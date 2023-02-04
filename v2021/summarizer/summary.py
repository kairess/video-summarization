import argparse

import torch
from transformers import ViTFeatureExtractor

from v2021 import extract_video_features

from .model import SummaryModel


class Summarizer:
    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.video_feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.model = SummaryModel.load_from_checkpoint(model_path).to(self.device)

    def summarize(self, video_path, threshold=0.7):
        video_features = extract_video_features(
            self.video_feature_extractor, video_path
        ).to(self.model.device)
        out = self.model(video_features)
        preds = (torch.sigmoid(out) > threshold).nonzero(as_tuple=True)[0]
        preds = preds.tolist()
        return preds


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--argument", help="Example argument")
    args = parser.parse_args()
