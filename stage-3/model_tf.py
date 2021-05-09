# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:02:12 2021

@author: albu
"""


# import tensorflow as tf
from tensorflow.keras import utils, callbacks, models, regularizers, Input, losses, metrics
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd


##################################
### chose what model you want to train: 'irrelevante_check', 'multitask'or 'quality'

what_to_train='multitask'
####################################




############################## Data Loader #############################

traindf = pd.read_csv("annotations_train.csv")

datagen=ImageDataGenerator(rescale=1./255.,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1)

test_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)
#use as class_type multi_output for multitask

def train_generator(images="image_name", y_true="irrelevant_image", class_type="raw"):
    train_gen=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="images/",
    x_col=images,
    y_col=y_true,
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode=class_type,
    target_size=(224,224))
    
    return train_gen

def test_generator(images="image_name", y_true="irrelevant_image", class_type="raw"):
    test_gen=test_datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="images/",
    x_col=images,
    y_col=y_true,
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode=class_type,
    target_size=(224,224))
    return test_gen

################################# Model ###############################

def model(lr=0.0001, input_shape=(224, 224, 3), base_model_trainable=False, model_name = 'irrelevant_vs_relevant'):
    
    
    base_model=MobileNetV2(weights='imagenet', input_shape=input_shape, include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(200, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation="relu")(x)
    
    if model_name == 'irrelevant_vs_relevant':
        out = Dense(1, activation="sigmoid", name='irrelevant')(x)
        loss = losses.binary_crossentropy
    
    elif model_name == 'multitask':
        out_width = Dense(1, activation="sigmoid", name='single_car')(x)
        
        # Changed this to 3, because, using softmax we need one neuron to each class
        # out_pavement = Dense(1, activation="softmax", name='pavement')(x)
        out_pavement = Dense(3, activation="softmax", name='pavement')(x)
        loss = [losses.binary_crossentropy, losses.categorical_crossentropy]
        
        out = [out_width, out_pavement]
        
    elif model_name == 'quality':
        out = Dense(3, activation='softmax', name='quality')(x)
        loss = losses.categorical_crossentropy
        
    model = Model(inputs = base_model.input, outputs = out)
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer, loss, metrics=[
        'accuracy',
        metrics.AUC(name='AUC'),
    ])

    return model


############################ Train ####################################

if what_to_train=='irrelevante_check':
    
    train_gen=train_generator()
    test_gen=test_generator()
    model=model(lr=0.0001)


elif what_to_train=='multitask':
    train_gen=train_generator(y_true=["irrelevant_image","single_car"],class_type="multi_output")
    test_gen=test_generator(y_true=["irrelevant_image","single_car"],class_type="multi_output")
    model=model(lr=0.0001 ,model_name = 'multitask')



#train cycle

STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=test_gen.n//test_gen.batch_size

history = model.fit_generator(generator=train_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=test_gen,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=50
)

y_hat= model.evaluate(test_gen)


