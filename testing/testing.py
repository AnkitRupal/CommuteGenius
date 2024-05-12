print("Importing Libraries", end = " \t")
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score
np.random.seed(42)

IMG_HEIGHT, IMG_WIDTH = 30, 30

print("Loading Neural Network Model", end = " \t")
# IMPORTING MODEL
modelPath = 'model.h5'
model=tf.keras.models.load_model(modelPath)

print("Loading Image Files", end = " \t")
# Testing
test = pd.read_csv('Test.csv')
labels = test["ClassId"].values
imgs = test["Path"].values
data_dir = "Test"
data =[]

for img in imgs:
    try:
        image = cv2.imread(img)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
        data.append(np.array(resize_image))
    except:
        print("Error in " + img)
X_test = np.array(data)
X_test = X_test/255

print("Start Predictions", end = " \t")
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

#Accuracy with the test data
print('Test Data accuracy: ',accuracy_score(labels, predicted_classes)*100)