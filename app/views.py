from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os 
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

'''Helper Functions'''

def get_label(pred_vector):
    """Turns pred vector into labels"""
    return ubreeds[np.argmax(pred_vector)]

def load_model(model_path):
    """Load a saves model"""
    print(f"Loading model from {model_path}")
    model=tf.keras.models.load_model(model_path,custom_objects={"KerasLayer":hub.KerasLayer},compile=False)
    return model

#Function to preprocess image
def preprocess(path):
    """CONVERT image into tensor"""
    img_size=224  
  #Reading an image file
    image=tf.io.read_file(path)
  #Turning image into numerical tensor with RGB color channels
    image=tf.image.decode_jpeg(image,channels=3)
  #Normalizing numerical RGB values (0-255) to (0-1)
    image=tf.image.convert_image_dtype(image,tf.float32)
  #Resizing image
    image=tf.image.resize(image,size=[img_size,img_size])
    return image

def create_batches(X):
    """Returns data batches,shuffles incase of training data,Takes only X incase of test data"""
  #for test data
    BATCH_SIZE=32
    print("Creating batches for test data...")
    #Slicing data
    data=tf.data.Dataset.from_tensor_slices((tf.constant(X)))
    #Preprocessing data and dividing into batches
    data_batch=data.map(preprocess).batch(BATCH_SIZE)
    return data_batch

def slicor(str):
    return str.replace('_',' ').upper()




labels_data=pd.read_csv('static/labels.csv')
ubreeds=np.unique(labels_data.breed)
model=load_model("static/model/20200817-15501597679424-full_model_mobilenetv2_adam.h5")

# Create your views here.
def home(request):
    if request.method=='POST':
        image=request.FILES.get('dog')
        if image is None:
            return render(request,'index.html')
        move("media/","static/cache/")
        fs=FileSystemStorage() #filestorage obj
        name=fs.save(image.name,image)
        path=os.path.join('media',name)
        #path=os.path.join(BASE_DIR,path)
        breed=predict([path])
        return render(request,'pred.html',{'imgurl':fs.url(name),'pred':breed})
    return render(request,'index.html')


def predict(path):
    image=create_batches(path)
    preds=model.predict(image)
    index=np.argmax(preds)
    breed=get_label(preds)
    pred=preds[0][index]
    print(pred)
    if pred<0.75:
        return "LoL! Its Not a Dog :P"
    breed=slicor(breed)
    return breed



def move(dir1,dir2):
    prefiles=os.listdir(dir2)       #Files already in cached
    for file in os.listdir(dir1):   #for each file in media
        if file in prefiles:        #if media has same file that is cached
            os.remove(dir1+file)    #remove it
            continue                #continue
        shutil.move(dir1+file,dir2) #move to cached