import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import cv2
import time
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib

print(torchvision.__version__)

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


edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6)
]


def main():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, num_keypoints=17)
    model.to(device).eval()

    score_threshold = 0.7

    transform = transforms.Compose([
        transforms.ToTensor()
    ])


    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        
        iteration_time = time.time()

        success, frame = cap.read()
        if not success:
            print("Webcam failed somehow?")
            continue


        model_input = transform(frame.astype(np.float32) / 255.0).unsqueeze(0).to(device)
        # print(model_input)
        with torch.no_grad():
            outputs = model(model_input)[0]


        boxes = outputs["boxes"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        pred_classes = [coco_names[i] for i in labels]

        scores = outputs["scores"].cpu().numpy()
        keypoints = outputs["keypoints"].cpu().numpy()
        keypoints_scores = outputs["keypoints_scores"].cpu().numpy()
        print("****************************************************")
        print(keypoints.shape)

        boxes = boxes[scores >= score_threshold].astype(np.int32)

        # exit()
        display = draw_keypoints(keypoints, scores, frame)

        # labels = outputs[0]['labels'].cpu().numpy()
        # pred_classes = [coco_names[i] for i in labels]
        # pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        # pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        # boxes = pred_bboxes[pred_scores >= score_threshold].astype(np.int32)

        frame = draw_boxes(boxes, pred_classes, labels, frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        print("iteration_time", time.time() - iteration_time)
    
    cap.release()


def draw_keypoints(pred_keypoints, scores, image):
    # the `outputs` is list which in-turn contains the dictionaries 
    print("num kp:", len(pred_keypoints))
    print(pred_keypoints)
    for i in range(len(pred_keypoints)):
        print("++++")
        print(pred_keypoints.shape)
        keypoints = pred_keypoints[i]
        # proceed to draw the lines if the confidence score is above 0.9
        if scores[i] > 0.9:
            keypoints = keypoints[:, :].reshape(-1, 3)
            for p in range(keypoints.shape[0]):
                # draw the keypoints
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 
                            3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                # uncomment the following lines if you want to put keypoint number
                # cv2.putText(image, f"{p}", (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # for ie, e in enumerate(edges):
            #     # get different colors for the edges
            #     rgb = matplotlib.colors.hsv_to_rgb([
            #         ie/float(len(edges)), 1.0, 1.0
            #     ])
            #     rgb = rgb*255
            #     # join the keypoint pairs to draw the skeletal structure
            #     keypoints = keypoints.astype(np.uint8)
            #     cv2.line(image, (keypoints[e, 0][0], keypoints[e, 1][0]),
            #             (keypoints[e, 0][1], keypoints[e, 1][1]),
            #             tuple(rgb), 2, lineType=cv2.LINE_AA)
        else:
            continue
    return image


def draw_boxes(boxes, classes, labels, image):

    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
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
