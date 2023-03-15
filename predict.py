
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
    print("Shape avant",np_img.shape)
    np_img = np_img.reshape(-1,512,384,3)
    print("Shape after",np_img.shape)
    return np_img





def predict(path_to_img,bin_model,multi_model):
    '''
    import the two models
    make predictions and
    from one image with one object, return the original image and the category predicted
    '''
    #prediction = model1.predict(preprocess(img))

    prediction = bin_model.predict(preprocess_binary(path_to_img))
    # [PROBA_BIODEGRADABLE,PROBA_NONBIODEGRADABLE]
    [[0.07,0.92]]
    if prediction.tolist()[0][0]>0.5 : # Then it's biodegradable : STOP here
        return "Biodegradable"

    # Sinon on lance le modele 2, YEAH !

    probas = multi_model.predict(preprocess_multiclass(path_to_img))  #ressemblera a (0.1,0.3,0.4,0.1,0.2)



    print(probas)
    #  max(probas)< 0.3
    # predicted_probabilities = model.predict(X_test)
    # predicted_probabilities.shape
    # y_pred = np.argmax(predicted_probabilities, axis = 1)
    # y_pred.shape

if __name__=="__main__":
    # preprocess_binary("image_loup.jpg")
    # print("Binary Preproc done")
    # preprocess_multiclass("image_loup.jpg")
    # model_binary_path = os.path.dirname(__file__)+'/model1_binary.h5'
    # model_binary = load_model(model_binary_path)
    model_multiclass_path = os.path.dirname(__file__)+'/models'
    model_multiclass = load(model_multiclass_path)
    print(type(model_multiclass))
    #predict("image_loup.jpg",model_binary,model_multiclass)
