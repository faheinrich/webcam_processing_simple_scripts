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

    # The output dictionary contains:
    # num_detections: a tf.int tensor with only one value, the number of detections [N].
    # detection_boxes: a tf.float32 tensor of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].
    # detection_classes: a tf.int tensor of shape [N] containing detection class index from the label file.
    # detection_scores: a tf.float32 tensor of shape [N] containing detection scores.
    # raw_detection_boxes: a tf.float32 tensor of shape [1, M, 4] containing decoded detection boxes without Non-Max suppression. M is the number of raw detections.
    # raw_detection_scores: a tf.float32 tensor of shape [1, M, 90] and contains class score logits for raw detection boxes. M is the number of raw detections.
    # detection_anchor_indices: a tf.float32 tensor of shape [N] and contains the anchor indices of the detections after NMS.
    # detection_multiclass_scores: a tf.float32 tensor of shape [1, N, 90] and contains class score distribution (including background) for detection boxes in the image including background class.

    # Apply image detector on a single image.
    detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

    score_threshold = 0.5

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
    
        iteration_time = time.time()

        success, frame = cap.read()
        if not success:
            print("Webcam failed somehow?")
            continue
        

        model_input = tf.convert_to_tensor(frame[None,...])
        detector_output = detector(model_input)

        print(detector_output)

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