import numpy as np
import cv2
from fastapi import FastAPI
import joblib
from predict import predict

app = FastAPI()
# Load models
binary_model = joblib.load("Path/to/the/binary_model.h5")
classification_model = joblib.load("Path/to/the/classification_model.h5")




@app.get("/")
def index():
    return {"Hello": "World"}

@app.post('/upload_image')
async def receive_image(img):#: UploadFile=File(...)): as it is
    '''
    Receiving and decoding the image
    '''
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    global Uploaded_Image
    Uploaded_Image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray


@app.get("/predict_one")
def predict_one():
    '''
    Takes the last image uploaded, return a prediction
    If it's biodegradable return B
    Else Return what type of waste it is
    '''
    # Import predict et display and return a prediction




# @app.get("/predict_all")
# def predict_all():
#     '''
#     Takes the last image uploaded, return a prediction of every objects inside
#     If it's biodegradable return B
#     Else Return what type of waste it is
#     '''

#     # from detect import detect
#     # image = detect(image)
#     #
