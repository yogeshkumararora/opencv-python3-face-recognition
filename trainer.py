import os
import cv2
import numpy as np
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create()
dataSetDirectory='dataSet'

def getImagesWithId(dataSetDirectory):
    imagePaths = [os.path.join(dataSetDirectory,f) for f in os.listdir(dataSetDirectory)]
    print(imagePaths)

    faces=[]
    Ids = []

    for imagePath in imagePaths:
        #Covert to gray scale (just to be double sure)
        faceImg = Image.open(imagePath).convert('L');
        faceNp = np.array(faceImg, 'uint8');
        Id = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        Ids.append(Id)

        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return np.array(Ids), faces

Ids, faces = getImagesWithId(dataSetDirectory)
recognizer.train(faces, Ids)
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()



getImagesWithId(dataSetDirectory)
