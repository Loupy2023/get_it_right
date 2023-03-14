import numpy as np
import cv2
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def index():
    return {"Hello": "World"}

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)): #as it is
    ### Receiving and decoding the image
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    Uploaded_Image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
