import torch
import cv2
import numpy
import time
import numpy as np


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



def main():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.eval().to(device)
    # model.classes = [0] # filter for specific classes

    confidenece_threshold = 0.7

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        
        iteration_time = time.time()

        success, frame = cap.read()
        if not success:
            print("Webcam failed somehow?")
            continue
        frame = frame[:,:,::-1].copy()

        model_output = model(frame)
        results = model_output.pandas().xyxy[0]
        pred_classes = results["name"]
        labels = results["class"]
        scores = results["confidence"]

        boxes = model_output.xyxy[0].cpu().numpy()[:,:4].astype(np.int32)

	# filter condidence thresh
        boxes = boxes[scores > confidenece_threshold]
        labels = labels[scores > confidenece_threshold]

        frame = draw_boxes(boxes, pred_classes, labels, frame)

        cv2.imshow('frame', frame[:,:,::-1])
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # model_output.print()
        print("iteration_time", time.time() - iteration_time)


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



if __name__ == "__main__":
    main()
