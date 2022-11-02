import pandas as pd
import keras
import numpy as np
from PIL import Image

import config as cnfg
import create_data as cd

from pdb import set_trace as bp

def load_compiled_model():
    # model = keras.models.load_model(cnfg.model_path)
    model = keras.models.load_model(r"C://Users//1//Desktop//אפרת לימודים 2022//BOOTCAMP//PROJECT//model_Anna_0_1_2_4_14_after_nirmul  ")
    return model

def load_history():
    history=pd.read_csv(r"C://Users//1//Desktop//אפרת לימודים 2022//BOOTCAMP//PROJECT//history_fnn_Anna__0_1_2_4_14_after_nirmul.csv")
    # print(history.head())
    return history

def load_data():
    loaded_data = np.load('./cfar10_modified_1000_no_food_container.npz')
    x_train = loaded_data['train'].astype('float32')/255
    x_validation = loaded_data['validation'].astype('float32')/255
    x_test = loaded_data['test'].astype('float32')/255
    y_train = loaded_data['ytrain']
    y_test = loaded_data['ytest']
    y_validation = loaded_data['yvalidation']
    num_classes = np.max(y_train) + 1
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_validation = keras.utils.to_categorical(y_validation, num_classes)
    return  x_train, x_validation, x_test, y_train, y_validation, y_test
# def predict():

# def convert_ypred3d_ypred1d():

def load_our_labels():
    DATA = pd.read_csv(cnfg.csv_path)
    value_count_dict = dict(DATA['label'].value_counts())
    value_count_dict = {k: value_count_dict[k] for k in sorted(value_count_dict)}
    labels_array = {label:cd.create_classes_dict()[label] for label in value_count_dict.keys()}
    print(cd.create_classes_dict())
    print(labels_array)
    return labels_array

def predict_by_image(image):
    print("predict_by_image")
    model = load_compiled_model()
    if isinstance(image, str):
        image = Image.open(image)
    image = np.resize(image,(32, 32,3))
    image = image.reshape(-1, 32, 32,3)
    image = image.astype('float32')
    image/=255
    prediction = model.predict(image)
    print(prediction)
    pred = np.argsort(prediction)
    print(pred)
    pred=pred[0][-3:]
    print(pred)
    labels=[cd.create_classes_dict()[pred[-1]],cd.create_classes_dict()[pred[-2]],cd.create_classes_dict()[pred[-3]]]
    percent=[ "%5.2f" %(float(prediction[0][pred[-1]])*100)+"%", "%5.2f" %(float(prediction[0][pred[-2]])*100)+"%","%5.2f" %(float(prediction[0][pred[-3]])*100)+"%"]
    dict1=dict()
    df=pd.DataFrame(dict1)
    for i in range(len(percent)):
        dict1.update({labels[i]:percent[i]})
    print(dict1)

    return dict1
# predict_by_image("Frog-PNG-3 (1).png)")

# def predict(img:np.ndarray):
#     model = load_compiled_model()
#     img = img.astype('float32')
#     img /= 255
#     img=img.reshape(-1,img.shape[0],img.shape[1],img.shape[2])
#     # label=model(img).numpy()[0]
#     label=model.predict(img)[0]
#     return load_our_labels()[label.argmax()]
#
# print(predict)