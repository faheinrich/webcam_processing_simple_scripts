import cv2
import numpy as np
import time
import torch

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def process_frame(predictor, cfg, frame):

    outputs = predictor(frame)
    v = Visualizer(frame, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    return out.get_image()


def main():

    model_yaml = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    score_threshold = 0.7
    cfg = get_cfg()   # get a fresh new config
    cfg.merge_from_file(model_zoo.get_config_file(model_yaml))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml)
    predictor = DefaultPredictor(cfg)

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        if not ret:
      	    print("webcam failed")
      	    break

        frame = process_frame(predictor, cfg, frame)
        
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == 27: # use ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
