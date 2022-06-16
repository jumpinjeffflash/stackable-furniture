import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
model = tf.keras.models.load_model('stackable_or_not.h5')

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.metrics import binary_accuracy
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from PIL import Image, ImageOps
import pandas as pd
import numpy as np
from cv2 import cv2

import random
import os

import streamlit as st

st.title('Furniture Stackability Classifier')

st.markdown("This dashboard takes pictures of furniture (click the box below for more details about the model) and predicts whether they're stackable.") 
st.markdown("Stackability was determined by the furniture type, the item's width/height/depth dimensions, along with some human logic when looking at the pictures (e.g. taking the shape of the item into account)")

with st.expander("Click here for more details about how this model was built"):
        st.write("""The is a Binary Classification model using a Convolutional Neural Network (CNN) to convert images into grids of numbers, which it then scans to discover patterns.""") 
        st.write("""Over 3,000 images were collected to train and test the model, comprising the following furniture types...""")
        st.write("""Stackable items: benches, chairs, coffee tables & stools""")
        st.write("""NON-Stackable items: armoires, cabinets/hutches, dressers, grandfather clocks, pianos & sofas/sectionals""")

@st.cache(persist=True)
        
def import_and_predict(image_data, model):
        size = (224,224)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)   
                
        img_reshape = image[np.newaxis,...]
    
        prediction = model.predict(img_reshape)

        return prediction
    
file = st.file_uploader("Please upload your furniture image...", type=["png","jpg","jpeg"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, width=100)
    prediction = import_and_predict(image, model)
      
    if prediction>0.50:
        st.write("""### This looks like it's stackable""")
    else:
        st.write("""### This doesn't look like it's stackable""")

    percentage = prediction*100
    out_arr = np.array_str(percentage, precision=2, suppress_small=True)
    
    probability = out_arr.strip("[").strip("]")
    probability_finessed = probability+"%"
  
    st.markdown("For context, here's our model's prediction of how stackable it is:")
    st.write(probability_finessed)
    st.markdown("(Scale: 100% = totally stackable | 0% = not at all stackable)")