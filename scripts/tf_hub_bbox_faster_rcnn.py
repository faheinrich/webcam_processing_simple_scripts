import cv2
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


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

# camera image
height = 480
width = 640



def main():

    # detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1")
    detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")

    score_threshold = 0.7

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
    
        iteration_time = time.time()

        success, frame = cap.read()
        if not success:
            print("Webcam failed somehow?")
            continue
        

        model_input = tf.convert_to_tensor(frame[None,...])
        detector_output = detector(model_input)

        pred_classes = detector_output["detection_classes"].numpy().astype(np.uint8)[0]
        pred_scores = detector_output["detection_scores"].numpy()[0]
        pred_bboxes = detector_output["detection_boxes"].numpy()[0]
        boxes = pred_bboxes[pred_scores >= score_threshold]
        named_classes = [coco_names[i] for i in pred_classes]

        frame = draw_boxes(boxes, named_classes, pred_classes, frame)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        print("iteration_time", time.time() - iteration_time)



def draw_boxes(boxes, classes, labels, image):

    for i, box in enumerate(boxes):
        ymin, xmin, ymax, xmax = box
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(xmin*width), int(ymin*height)),
            (int(xmax*width), int(ymax*height)),
            color, 2
        )
        cv2.putText(image, classes[i], (int(xmin*width), int(ymin*width-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image



if __name__ == "__main__":
    main()
