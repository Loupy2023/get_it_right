

def preprocess_binary(img):
    '''
    Transform an image so it can be passed to the binary model
    return np.array with the correct dimensions
    '''
    pass




def preprocess_multiclass(img):
    '''
        Transform an image so it can be passed to the multiclass model
    return np.array with the correct dimensions
    '''
    pass



def predict(img):
    '''
    import the two models
    make predictions and
    from one image with one object, return the original image and the category predicted
    '''
    #prediction = model1.predict(preprocess(img))

    # if prediction == ecyclable  : return prediction

    # probas = model2.predict_proba(preprocess_multiclass(img))  ressemblera a (0.1,0.3,0.4,0.1,0.2)
    #  max(probas)< 0.3
