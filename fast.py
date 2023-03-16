import os
import pandas as pd
import numpy as np
import cv2
from fastapi import FastAPI,File
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

from predict import predict

app = FastAPI()

model_binary_path = os.path.dirname(__file__)+'/model1_binary.h5'
app.state.model_binary = load_model(model_binary_path)

model_multiclass_path = os.path.dirname(__file__)+'/multiclass.h5'
app.state.model_multiclass = load_model(model_multiclass_path)

# Optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return dict(greeting="Is alive!")

@app.post('/upload_image')
async def receive_image(file: bytes = File(...)):#: UploadFile=File(...)): as it is
    '''
    Receiving and decoding the image
    '''

    with open('data/image.jpg','wb') as image:
        image.write(file)
        image.close()
    return 'got it'


@app.get("/predict_one")
def predict_one():
    '''
    Takes the last image uploaded, return a prediction
    If it's biodegradable return B
    Else Return what type of waste it is
    '''
    # Import predict et display and return a prediction
    img= 'data/image.jpg'

    return predict( img
                    ,app.state.model_binary
                    ,app.state.model_multiclass)
