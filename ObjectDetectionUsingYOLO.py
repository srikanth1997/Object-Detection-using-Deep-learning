import numpy as np
import os
import argparse
import time
import cv2
# import imutils
# from imutils import paths

class ObjectDetectionUsingYOLO:

    inptArgs={}
    weightsLoc=''
    configLoc=''
    colorPalette=[]
    labels=[]
    finalBestBoundingBoxes=None
    image=None
    images = []
    classId=[]
    predictedBoundingBoxes=[]
    confidenceScores = []
    imagePaths = []

    def parseInputArgs():
        ap = argparse.ArgumentParser()
        # ap.add_argument("-i", "--image", required=True,help="points to image directory")
        ap.add_argument("-y", "--yolo", required=True,help = "to yolo main direc")
        ap.add_argument("-c", "--confidence", type = float, default= 0.5,help = "min value of confidence, which can be changed")
        ap.add_argument("-t", "--threshold", type= float, default= 0.3,help="min threshold value")
        ObjectDetectionUsingYOLO.inptArgs = vars(ap.parse_args())



    def initialize():
        print("Initializing the required data structures for YOLO\n")
        labelsLoc = os.path.sep.join([ObjectDetectionUsingYOLO.inptArgs["yolo"], "coco.names"])
        path = os.getcwd()+str("\images")
        ObjectDetectionUsingYOLO.imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        # for i in imagePaths[:] :
        #     tmpImage = cv2.imread(i)
        #     tmpImage = cv2.resize(tmpImage, (416,416))
        #     ObjectDetectionUsingYOLO.images.append(tmpImage)
        ObjectDetectionUsingYOLO.labels = open(labelsLoc).read().strip().split("\n")
        np.random.seed(42)
        ObjectDetectionUsingYOLO.colorPalette = np.random.randint(0,255, size=(len(ObjectDetectionUsingYOLO.labels), 3), dtype="uint8")
        ObjectDetectionUsingYOLO.weightsLoc = os.path.sep.join([ObjectDetectionUsingYOLO.inptArgs["yolo"], "yolov3.weights"])
        ObjectDetectionUsingYOLO.configLoc = os.path.sep.join([ObjectDetectionUsingYOLO.inptArgs["yolo"], "yolov3.cfg"])



    def performDetection():
        print("Object detection process starts\n")
        model  = cv2.dnn.readNetFromDarknet(ObjectDetectionUsingYOLO.configLoc, ObjectDetectionUsingYOLO.weightsLoc)
        lnames = model.getLayerNames()
        lnames = [lnames[i[0]-1] for i in model.getUnconnectedOutLayers()  ]
        startTime = time.time()
        for (k,p) in enumerate(ObjectDetectionUsingYOLO.imagePaths):
            ObjectDetectionUsingYOLO.predictedBoundingBoxes= []
            ObjectDetectionUsingYOLO.confidenceScores = []
            ObjectDetectionUsingYOLO.classId = []
            ObjectDetectionUsingYOLO.image = cv2.imread(p)
            (ht, wt) = ObjectDetectionUsingYOLO.image.shape[:2]
            #getting the output layer names from yolo
            blob = cv2.dnn.blobFromImage(ObjectDetectionUsingYOLO.image, 1/255.0, (416,416),swapRB=True, crop=False )
            model.setInput(blob)
            layerOutputs = model.forward(lnames)
            #populating data from YOlO to the above lists
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence>ObjectDetectionUsingYOLO.inptArgs["confidence"]:
                        #yolo return x,y center corrdinates of bounding box.
                        #so scaling it back to initial img size
                        box = detection[0:4] * np.array([wt, ht, wt, ht])
                        (cX,cY, width, height) = box.astype("int")
                        #top left (x,y) coordinates
                        x = int(cX - (width/2))
                        y = int(cY - (height/2))
                        ObjectDetectionUsingYOLO.predictedBoundingBoxes.append([x, y, int(width), int(height)])
                        ObjectDetectionUsingYOLO.confidenceScores.append(float(confidence))
                        ObjectDetectionUsingYOLO.classId.append(classID)
            #applying non maxima suppression
            ObjectDetectionUsingYOLO.finalBestBoundingBoxes = cv2.dnn.NMSBoxes(ObjectDetectionUsingYOLO.predictedBoundingBoxes, ObjectDetectionUsingYOLO.confidenceScores,ObjectDetectionUsingYOLO.inptArgs["confidence"], ObjectDetectionUsingYOLO.inptArgs["threshold"])
            ObjectDetectionUsingYOLO.plotTheBoundingBoxes()
            name = os.path.basename(ObjectDetectionUsingYOLO.imagePaths[k])
            path = os.getcwd()+str("\output")+str("\\{}".format(name))
            cv2.imwrite(path, ObjectDetectionUsingYOLO.image)
        endTime = time.time()
        print("YOLO took {:.6f} seconds".format(endTime-startTime))
        print("Object detection completed\n")



    def plotTheBoundingBoxes():
        if len(ObjectDetectionUsingYOLO.finalBestBoundingBoxes) > 0:
            for i in ObjectDetectionUsingYOLO.finalBestBoundingBoxes.flatten():
                (x,y) = (ObjectDetectionUsingYOLO.predictedBoundingBoxes[i][0], ObjectDetectionUsingYOLO.predictedBoundingBoxes[i][1])
                (w,h) = (ObjectDetectionUsingYOLO.predictedBoundingBoxes[i][2], ObjectDetectionUsingYOLO.predictedBoundingBoxes[i][3])

                color = [int (c) for c in ObjectDetectionUsingYOLO.colorPalette[ObjectDetectionUsingYOLO.classId[i]]]
                cv2.rectangle(ObjectDetectionUsingYOLO.image, (x,y), (x+w, y+h), color, 2)
                text  = "{}: {:.4f}".format(ObjectDetectionUsingYOLO.labels[ObjectDetectionUsingYOLO.classId[i]], ObjectDetectionUsingYOLO.confidenceScores[i])
                cv2.putText(ObjectDetectionUsingYOLO.image, text, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # cv2.imshow("Image", image)
                # cv2.waitKey(0)
        # cv2.imshow("Image", ObjectDetectionUsingYOLO.image)
        # cv2.waitKey(0)


    def process():
        print("YOLO process started\n")
        ObjectDetectionUsingYOLO.parseInputArgs()
        ObjectDetectionUsingYOLO.initialize()
        ObjectDetectionUsingYOLO.performDetection()
        # ObjectDetectionUsingYOLO.plotTheBoundingBoxes()



ObjectDetectionUsingYOLO.process()
