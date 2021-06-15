
import numpy as np
import cv2
import time
import tensorflow as tf
import tensorflow_hub as hub

def main():
   
    input_size = 224 # efficientnetB0

    # create model
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/classification/1")
    ])
    model.build([None, input_size, input_size, 3])  # Batch input shape.

    # model.summary()

    # get imagenet labels
    labels_file = open('ImageNetLabels.txt', 'r')
    labels_raw = labels_file.readlines()
    imagenet_labels = [i.strip() for i in labels_raw[1:]]


    cap = cv2.VideoCapture(0)
    while cap.isOpened():

        process_time = time.time()

        # take webcam frame 
        success, frame = cap.read()
        if not success:
            print("webcam failed")
            continue

        # flip to rgb because cv2
        frame = frame[:,:,::-1]

        # model input
        cut_to_square = 80
        square_frame = frame[:,cut_to_square:-cut_to_square]
        input_image = cv2.resize(square_frame, (input_size, input_size))
        model_input = tf.expand_dims(input_image / 255, axis=0)

        # prediction
        preds = model.predict(model_input)[0]
        
        # print classified labels
        num_print_labels = 10
        sorted_inds = np.argsort(preds)[::-1][:num_print_labels]
        labels = [imagenet_labels[i] for i in sorted_inds]
        for i in labels:
            print(i)

        # show processed image
        frame = input_image
        cv2.imshow('webcam', cv2.resize(frame[:,:,::-1],(frame.shape[1]*2, frame.shape[0]*2)))
        if cv2.waitKey(5) & 0xFF == 27:
            break

        print("time:", time.time() - process_time)
        print("+++++++++++++++++")

    cap.release()


if __name__ == "__main__":
    main()