import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score
np.random.seed(42)

class Tester:
    def __init__(self) -> None:
        self.initialiseParameters()
        self.importModel('model.h5')
        self.imgFiles = self.readImageFiles()
        self.makePredictions(self.imgFiles)
        return 
    

    def initialiseParameters(self) -> None:
        self.__print__("Initialising Parameters")
        self.IMG_HEIGHT, self.IMG_WIDTH = 32, 32


    def importModel(self, modelName) -> None:
        self.__print__("Importing Model")
        self.model=tf.keras.models.load_model(modelName)
        return
    

    def readImageFiles(self) -> None:
        self.__print__("Loading Images")
        test = pd.read_csv('Test.csv')
        self.labels, imgs, data = test["ClassId"].values, test["Path"].values, []
        for img in imgs:
            try:
                image = cv2.imread(img)
                image_fromarray = Image.fromarray(image, 'RGB')
                resize_image = image_fromarray.resize((self.IMG_HEIGHT, self.IMG_WIDTH))
                data.append(np.array(resize_image))
            except:
                print("Error in " + img)
        X_test = np.array(data)
        X_test = X_test/255
        return X_test
    

    def makePredictions(self, X_test) -> None:
        self.__print__("Making Predictions")
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        print('Test Data accuracy: ',accuracy_score(self.labels, predicted_classes)*100)
        return 


    def __print__(self, message) -> None:
        print(message)
        return 


if __name__ == '__main__':
    Tester()