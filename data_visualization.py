import create_data as cd
import config as cnfg
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt


def show_10_image_of_class(class_label:int, cifar:str) -> None:
    """

    :param class_label:
    :param cifar:
    """
    DATA = pd.read_csv(cnfg.csv_path)
    if cifar == cnfg.cifar100:
        class_label = class_label+cnfg.num_classes_cifar10
    image_pathes=DATA[DATA['label']==class_label][:10]
    fig = plt.figure(figsize=(20, 14))

    for row in range(len(image_pathes)):
        fig.add_subplot(2, 5, row+1)
        img= cv2.imread(image_pathes.iloc[row]['image_path']+image_pathes.iloc[row]['image_name'])
        plt.imshow(img)
        plt.axis('off')
        plt.title(image_pathes.iloc[row]['image_name'])
    plt.tight_layout()
    plt.show()

def show_classes_count():
    DATA = pd.read_csv(cnfg.csv_path)
    value_count_dict=dict(DATA['label'].value_counts())
    value_count_dict={k:value_count_dict[k] for k in sorted(value_count_dict)}
    plt.figure(figsize=(8, 10))

    labels_array = cd.create_classes_dict()
    plt.bar(value_count_dict.keys(), value_count_dict.values(), color='pink',  width=0.4,bottom=0.15)
    plt.xticks(np.arange(len(labels_array)),labels_array.values(),rotation='vertical')

    plt.xlabel("classes")
    plt.ylabel("sum of images")
    plt.title("sum of images for each class")
    plt.tight_layout()
    plt.show()


def show_splited_classes_count():
    DATA = pd.read_csv(cnfg.csv_path)
    value_count_dict = dict(DATA['label'].value_counts())
    value_count_dict = {k: value_count_dict[k] for k in sorted(value_count_dict)}
    x_train, x_validation, x_test, y_train, y_validation, y_test=cd.split_train_test_validation()
    train=y_train.value_counts()
    train = {k: train[k] for k in sorted(train.keys())}
    validation = y_validation.value_counts()
    validation = {k: validation[k] for k in sorted(validation.keys())}
    test = y_test.value_counts()
    test = {k: test[k] for k in sorted(test.keys())}
    labels_array=[cd.create_classes_dict()[label] for label in value_count_dict.keys()]
    df = pd.DataFrame({"train": train,  "validation": validation, "test":test})
    ax = df.plot.bar(rot=0)
    plt.xticks(np.arange(len(value_count_dict.keys())), labels_array, rotation='vertical')
    plt.tight_layout()
    plt.show()

