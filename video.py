import numpy as np
import argparse
import imutils
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video file")
ap.add_argument("-m", "--mask-rcnn", required=True, help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="minimum threshold for pixel-wise mask segmentation")
ap.add_argument("-e", "--mask_rcnn_coco", required=True, help="Path to Mask R-CNN COCO dataset directory")
args = vars(ap.parse_args())
print(args)

labelsPath = os.path.sep.join([args["mask_rcnn_coco"], "object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = os.path.sep.join([args["mask_rcnn_coco"], "frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn_coco"], "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

vs = cv2.VideoCapture(args["input"])

frame_width=int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))


while True:

    (grabbed, frame) = vs.read()

    if not grabbed:
        break
    starting_time = time.time()
    
    resized = cv2.resize(frame,(1080,800))
    blob = cv2.dnn.blobFromImage(resized, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

    for i in range(0, boxes.shape[2]):

        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > args["confidence"]:
        
            (H, W) = resized.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
            mask = (mask > args["threshold"])

            roi = resized[startY:endY, startX:endX][mask]

            color = COLORS[classID]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            resized[startY:endY, startX:endX][mask] = blended

            color = [int(c) for c in color]
            cv2.rectangle(resized, (startX, startY), (endX, endY), color, 2)

            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(resized, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    ending_time = time.time()
    inference_time = ending_time - starting_time
    inference_time_ms = inference_time*1000 

    fps = 1/inference_time

    cv2.putText(resized,f'inference_time: {inference_time_ms:.2f}ms' , (25,230),cv2.FONT_HERSHEY_DUPLEX,0.6,(125,220,3),1)      
    cv2.putText(resized,f'FPS: {fps:.2f}' , (85,200),cv2.FONT_HERSHEY_DUPLEX,0.6,(125,670,3),1)      
    cv2.imshow("Frame", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()
