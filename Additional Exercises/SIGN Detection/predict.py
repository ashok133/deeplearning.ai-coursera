import math
import numpy as np
import h5py
import pickle
import scipy
import base64
import io
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from scipy import ndimage
from tensorflow.python.framework import ops
from utils import *

def img_to_base64(image_loc):
    with open(image_loc, "rb") as image_file:
        image_read = image_file.read()
        encoded_image = base64.encodestring(image_read)
        # encoded_image = base64.b64encode(image_file.read())
        # encoded_image = base64.b64encode(bytes('your string', 'utf-8'))
    save_and_predict(encoded_image)

def save_and_predict(img64):
    temp_rand = np.random.randn()
    file_loc = "sample_images/" + str(temp_rand) + ".png"
    # imgdata = base64.b64decode(img64)
    with open(file_loc, "wb") as fp:
        fp.write(base64.b64decode(img64))
    prediction = predict_sign(file_loc)
    return prediction

def predict_sign(image_loc):
    image = plt.imread(image_loc, 0)
    print(type(image))
    with open('learned_model/params.pickle', 'rb') as handle:
        params = pickle.load(handle)
        my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
        my_image_prediction = predict(my_image, params)
        plt.imshow(image)
        print("The network predicts a sign of :  " + str(np.squeeze(my_image_prediction)))
        return str(np.squeeze(my_image_prediction))
#
# def predict_sign_from_base_64(img_base64):
#     image = np.array(ndimage.imread(image_loc, flatten=False))
#     with open('learned_model/params.pickle', 'rb') as handle:
#         params = pickle.load(handle)
#         my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
#         my_image_prediction = predict(my_image, params)
#         plt.imshow(image)
#         print("The network predicts a sign of :  " + str(np.squeeze(my_image_prediction)))
#         return str(np.squeeze(my_image_prediction))

# predict_sign('sample_images/two.jpg')

img_to_base64('sample_images/two.jpg')
