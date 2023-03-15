import os
import pandas as pd
from fastapi import FastAPI
from tensorflow.keras.models import load_model
from tensorflow.saved_model import load
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

model_binary = os.path.dirname(__file__)+'/model1_binary.h5'
model1 = load_model(model_binary)

model_multiclass = os.path.dirname(__file__)+'/detection_model_by_fbl.h5'
model2 = load(model_multiclass)

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
