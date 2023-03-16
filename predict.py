
import os
from PIL import Image
import  numpy as np
from tensorflow.keras.models import load_model
from tensorflow.saved_model import load

def preprocess_binary(path_to_img):
    '''
    Transform an image so it can be passed to the binary model
    return np.array with the correct dimensions
    '''
    image = Image.open(path_to_img)
    image = image.resize((200,200))
    np_img = np.array(image)
    np_img = np_img.reshape(1,200,200,3)
    return np_img


def preprocess_multiclass(path_to_img):
    '''
        Transform an image so it can be passed to the multiclass model
    return np.array with the correct dimensions (512, 384)
    '''
    image = Image.open(path_to_img)
    new_image = image.resize((384, 512))
    np_img = np.array(new_image)
    #print("Shape avant",np_img.shape)
    np_img = np_img.reshape(-1,512,384,3)
    #print("Shape after",np_img.shape)
    return np_img


def predict(path_to_img,bin_model,multi_model):
    '''
    import the two models
    make predictions and
    from one image with one object, return the original image and the category predicted
    '''
    prediction_1 = bin_model.predict(preprocess_binary(path_to_img))

    if prediction_1.tolist()[0][0]>0.5 : # Then it's biodegradable : STOP here
        return "Biodegradable"


    # Sinon on lance le modele 2, YEAH !

    prediction_2 = multi_model.predict(preprocess_multiclass(path_to_img))  #ressemblera a np.array(0.1,0.3,0.4,0.1,0.2)

    labels_list = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic']
    if np.max(prediction_2)< 0.25 : return "Thrash"
    y_pred = labels_list[np.argmax(prediction_2[0,:])]
    return y_pred

if __name__=="__main__":
    preprocess_binary("image_loup.jpg")
    print("Binary Preproc done")
    preprocess_multiclass("image_loup.jpg")
    print("Multiclass Preproc done")
    model_binary_path = os.path.dirname(__file__)+'/model1_binary.h5'
    model_binary = load_model(model_binary_path)
    print("Model 1 loaded")
    model_multiclass_path = os.path.dirname(__file__)+'/multiclass.h5'
    model_multiclass = load_model(model_multiclass_path)
    print(type(model_multiclass))
    print(predict("image_loup.jpg",model_binary,model_multiclass))
