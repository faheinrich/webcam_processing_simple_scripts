import sys
from threading import Thread
import time

import numpy as np
import cv2
import torch

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QApplication, QPushButton, QLabel
from PyQt6.QtGui import QImage, QPixmap


coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def draw_boxes(boxes, classes, labels, image):

    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image


def pixmap_from_cv_image(cv_image: np.ndarray):
    print(type(cv_image))
    height, width, _ = cv_image.shape
    bytesPerLine = 3 * width
    qImg = QImage(cv_image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888).rgbSwapped()
    return QPixmap(qImg)


class DetectorGui(QMainWindow):

    # def keyPressEvent(self, event):
    #     print("Key pressed")

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Detection")

        width = 1300
        height = 800
        self.setMinimumSize(QSize(width, height))

        img = np.zeros((200, 200, 3))
        self.image_display = QLabel(self)
        # pic.setPixmap(QPixmap("Q107.png"))
        self.image_display.setPixmap(pixmap_from_cv_image(img))
        self.image_display.show()  # You were missing this.

        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.exit_btn)
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(exit_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_display)
        main_layout.addLayout(buttons_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.detector_thread = ObjectDetectionThread(self)
        self.detector_thread.start()

    def exit_btn(self):
        print("Exiting...")
        sys.exit()

    def change_img(self, img):
        self.image_display.setPixmap(pixmap_from_cv_image(img))


class ObjectDetectionThread(Thread):
    def __init__(self, gui: DetectorGui):
        Thread.__init__(self)
        self.daemon = True
        self.gui = gui

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print("Using device", self.device)

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval().to(self.device)
        self.confidenece_threshold = 0.7

        self.video_capture = cv2.VideoCapture(0)

    def run(self):
        time.sleep(0.1)
        while True:
            iteration_time = time.time()

            success, frame = self.video_capture.read()
            if not success:
                print("Webcam failed somehow?")
                continue
            frame = frame[:, :, ::-1].copy()

            model_output = self.model(frame)
            results = model_output.pandas().xyxy[0]
            pred_classes = results["name"]
            labels = results["class"]
            scores = results["confidence"]

            boxes = model_output.xyxy[0].cpu().numpy()[:, :4].astype(np.int32)

            boxes = boxes[scores > self.confidenece_threshold]
            labels = labels[scores > self.confidenece_threshold]

            img = draw_boxes(boxes, pred_classes, labels, frame)

            duration = time.time() - iteration_time
            print("iteration_time", duration, "FPS:", 1/duration)

            self.gui.change_img(img[:, :, ::-1].copy())


def main():
    app = QApplication(sys.argv)
    window = DetectorGui()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

