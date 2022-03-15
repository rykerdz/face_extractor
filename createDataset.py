from email.mime import base
import cv2
import math
import os
import numpy as np

base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + 'model_data/weights.caffemodel')

model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Getting video files
directory = os.fsencode(base_dir + "videos")
padding_y = 100 # Padding to crop the whole face 
padding_x = 75
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    cap = cv2.VideoCapture(base_dir + "videos/"+ filename)
    frameRate = cap.get(5) #frame rate
    x=1
    printt = True
    while(cap.isOpened()):
        print("here")
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            if printt == True:
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                model.setInput(blob)
                detections = model.forward()

                # Identify each face
                for i in range(0, detections.shape[2]):
                    
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    confidence = detections[0, 0, i, 2]

                    # If confidence > 0.5, save it as a separate file
                    if (confidence > 0.5):
                        frame2 = frame[startY-padding_y:endY+padding_y, startX-padding_x:endX+padding_x]
                        if frame.any():
                            print("x is "+str(x))
                            x+=1
                            cv2.imwrite(base_dir + 'results/' + str(x) + '_'+ filename + '.png', cv2.resize(frame2, (224, 224)))


                printt = False
            else:
                printt = True
    cap.release()
    print ("Done!")


# Capturing images from video

