from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2 
import os 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

print("Loading model...")
# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("./labels.txt", "r").readlines()

def readFileNames(path):
    fileNames = os.listdir(path)
    names = []
    for file in fileNames:
        names.append( path+"/"+file )
        #print( path+"/"+file)
    return names 

def readImages(fileNames): 
    imageNumber = len(fileNames)
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(imageNumber, 224, 224, 3), dtype=np.float32)

    # Default size trained images 
    size = (224, 224)
    
    imgPos = 0
    for file in fileNames:
        image = Image.open(file).convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center    
        #image = ImageOps.fit(image, size, Image.Resampling.LANCZOS) # For pillow version > 9.0.0 
        image = ImageOps.fit(image, size, Image.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[imgPos] = normalized_image_array
        imgPos = imgPos + 1 
    return data 


def showPrediction(dataImages,y):
    # Predicts the model
    #print(len(dataImages))
    prediction = model.predict(dataImages)
    yPred = []
    #print(prediction) 
    for p in prediction:
        index = np.argmax(p)
        #print(index) 
        yPred.append(index) 
    
    print("Accuracy: ", accuracy_score(y,yPred)*100, "%"  )
    cm = confusion_matrix(y, yPred)
    print("Confusion Matrix: ")
    print(cm)
    return yPred 


# Especifica a pasta que contém os arquivos de imagem
imageFolderApples = "./images/apples"
imageFolderTomatoes = "./images/tomatoes" 

print("Loading images...")
#Getting file names for apples 
applesFileNames = readFileNames(imageFolderApples)
 
#Getting classes for apples 
y0 = np.ones(len(applesFileNames), dtype=int)*0 # 0 - apple class

#Geting file names for tomatoes  
tomatoesFileNames = readFileNames(imageFolderTomatoes)
 
#Getting classes for tomatoes 
y1 = np.ones(len(tomatoesFileNames), dtype=int)*1 # 1 - tomatoes class


yTest =  np.concatenate([y0,  y1]  )

fileNames = applesFileNames + tomatoesFileNames
#print(fileNames)

dataImgs = readImages(fileNames) 

print("Show prediction results...")   
yp = showPrediction(dataImgs,yTest)  

selArray = yp != yTest
print("Failed image predictions: ")
cont = 0 
for c in selArray:     
    if c: 
        print(fileNames[cont])
    cont = cont + 1 


