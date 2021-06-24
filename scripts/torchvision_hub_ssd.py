import torch
import numpy as np
import cv2
import time

import torchvision.transforms as transforms 

import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
][1:]


def main():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
    ssd_model.eval().to(device)
    
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    uris = [
    'http://images.cocodataset.org/val2017/000000397133.jpg',
    'http://images.cocodataset.org/val2017/000000037777.jpg',
    'http://images.cocodataset.org/val2017/000000252219.jpg'
    ]

    inputs = [utils.prepare_input(uri) for uri in uris]
    tensor = utils.prepare_tensor(inputs)

    with torch.no_grad():
        detections_batch = ssd_model(tensor)

    

    results_per_input = utils.decode_results(detections_batch)
    best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]
    print(best_results_per_input)
    
    exit()


    




    # for image_idx in range(len(best_results_per_input)):
    #     fig, ax = plt.subplots(1)
    #     # Show original, denormalized image...
    #     image = inputs[image_idx] / 2 + 0.5
    #     ax.imshow(image)
    #     # ...with detections
    #     bboxes, classes, confidences = best_results_per_input[image_idx]
    #     for idx in range(len(bboxes)):
    #         left, bot, right, top = bboxes[idx]
    #         x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
    #         rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    #         ax.add_patch(rect)
    #         ax.text(x, y, "{} {:.0f}%".format(coco_names[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
    # plt.show()


    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        iteration_time = time.time()

        success, frame = cap.read()
        if not success:
            print("Webcam failed somehow?")
            continue
        frame = frame[:,:,::-1].copy()

        tensor = transform(frame)[None, ...]
        tensor = utils.prepare_tensor(tensor)

        with torch.no_grad():
            detections_batch = ssd_model(tensor)
        
        results_per_input = utils.decode_results(detections_batch)
        best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]

        print(best)

    
        
        
        cv2.imshow('frame', frame[:,:,::-1])
        if cv2.waitKey(1) & 0xFF == 27:
            break

        print("iteration_time", time.time() - iteration_time)
    cap.release()



if __name__ == "__main__":
    main()