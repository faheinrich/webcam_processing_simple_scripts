import time
import sys
import numpy as np
import cv2
import torch
import torchvision

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



def main():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(torch.cuda.get_device_name(device))

    if len(sys.argv) > 1:
        if sys.argv[1] == "fast":
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, pretrained_backbone=True)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    
    model.eval().to(device)

    score_threshold = 0.65

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])


    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        
        iteration_time = time.time()

        success, frame = cap.read()
        if not success:
            print("Webcam failed somehow?")
            continue
    
        frame = frame[:,:,::-1].copy() / 255
        frame = frame.astype(np.float32)
        model_input = transform(frame).unsqueeze(0).to(device)
        outputs = model(model_input)

        labels = outputs[0]['labels'].cpu().numpy()
        pred_classes = [coco_names[i] for i in labels]
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        boxes = pred_bboxes[pred_scores >= score_threshold].astype(np.int32)

        frame = draw_boxes(boxes, pred_classes, labels, frame)

        cv2.imshow('frame', cv2.resize(np.uint8(frame[:,:,::-1] * 255), (frame.shape[1]*3, frame.shape[0]*3)))
        if cv2.waitKey(1) & 0xFF == 27:
            break

        print("iter_time:", "%.4f" % (time.time() - iteration_time), "hz:", "%.2f" % (1/(time.time() - iteration_time)))
    
    cap.release()



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
