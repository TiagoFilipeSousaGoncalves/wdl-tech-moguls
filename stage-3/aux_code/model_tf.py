
import tensorflow as tf
from tensorflow.keras import utils, callbacks, models, regularizers, Input, losses, metrics
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image

import pandas as pd
import numpy as np
import math

##################################
# chose what model you want to train: 'irrelevante_check', 'multitask' or 'quality'

what_to_train='multitask'
EPOCHS = 100
batch_size=32
LR=0.0001

############################## Data Loader #############################

#read dataframe
df = pd.read_csv("annotations.csv")

# new = df["image"]= df["image"].str.split("/", n = 3, expand = True)
# df["image"] = new[3]
df["image"] = df["image"].str.split("/", n = 3, expand = True)[3]

df['irrelevant_infer'] = (df['street_width']=='irrelevant_image')*1

#change dataframe for every specific task
if what_to_train=='quality':
    df = df[df['pavement_quality'].notna()]

elif what_to_train=='multitask':
    df = df[df['pavement_type'].notna()]
    df['pavement_type'][df['pavement_type']=='alcatrao']=0
    df['pavement_type'][df['pavement_type']=='terra_batida']=1
    df['pavement_type'][df['pavement_type']=='paralelo']=2
    df['street_width'][df['street_width']=='single_car']=0
    df['street_width'][df['street_width']=='double_car_or_more']=1
    




#Train vs test division of dataframe by group of images 
df = df[df['image'].notna()]
df['image_group'] = df.image.str.extract(r"(image\d{1,})")

train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state =42).split(df, groups=df['image_group']))
train = df.iloc[train_inds]
test = df.iloc[test_inds]

#images generator for multitask 
def multitask_generator(data, batch_size=32):
        imagePath = "images/"
        # imagePath = "data/Competition/images/"

        swID = len(data.street_width.unique())
        ptID = len(data.pavement_type.unique())
        images, sws,pts = [], [], []
        while True:
            for i in range(0, data.shape[0]):
                r = data.iloc[i]
                name, sw, pt = r['image'], r['street_width'], r['pavement_type']
                im = Image.open(imagePath+name)
                im = im.resize((224, 224))
                im = np.array(im) / 255.0
                images.append(im)
                sws.append(tf.keras.utils.to_categorical(sw, swID))
                pts.append(tf.keras.utils.to_categorical(pt, ptID))
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(sws), np.array(pts)]
                    images, sws, pts = [], [], []


datagen=ImageDataGenerator(rescale=1./255.,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1)

test_datagen = ImageDataGenerator(rescale=1./255)


def train_generator(images="image", y_true="irrelevant_infer", class_type="raw"):
    train_gen=datagen.flow_from_dataframe(
    dataframe=train,
    directory="images/",
    x_col=images,
    y_col=y_true,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode=class_type,
    target_size=(224,224))
    
    return train_gen

def test_generator(images="image", y_true="irrelevant_infer", class_type="raw", BS=32):
    test_gen=test_datagen.flow_from_dataframe(
    dataframe=test,
    directory="images/",
    x_col=images,
    y_col=y_true,
    batch_size=BS,
    seed=42,
    shuffle=False,
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
        out_width = Dense(1, activation="sigmoid", name='number_cars')(x)
        out_pavement = Dense(3, activation="softmax", name='pavement')(x)
        loss = [losses.binary_crossentropy, losses.categorical_crossentropy]
        
        out = [out_width, out_pavement]
        
    elif model_name == 'quality':
        out = Dense(3, activation='softmax', name='quality')(x)
        loss = losses.categorical_crossentropy
        
    model = Model(inputs = base_model.input, outputs = out)
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer, loss, metrics=['accuracy'])

    return model


############################ Train ####################################

if what_to_train=='irrelevante_check':
    train_gen=train_generator()
    test_gen=test_generator()
    model=model(lr=LR)
    monitor_check='val_loss'


elif what_to_train=='multitask':
    train_gen  = multitask_generator(train, batch_size=batch_size)
    test_gen  = multitask_generator(test, batch_size=batch_size)
    model=model(lr=LR ,model_name = 'multitask')
    monitor_check='val_pavement_loss'
    
elif what_to_train=='quality':
    train_gen=train_generator(y_true="pavement_quality",class_type="categorical")
    test_gen=test_generator(y_true="pavement_quality",class_type="categorical")
    model=model(lr=LR, model_name ='quality')
    monitor_check='val_loss'


#train cycle

STEP_SIZE_TRAIN=len(train)//batch_size
STEP_SIZE_VALID=len(test)//batch_size



checkpoint_filepath = what_to_train+'.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor=monitor_check,
    mode='min',
    save_best_only=True,
    verbose=1)

history = model.fit_generator(generator=train_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=test_gen,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint_callback]
)



########### Test ##############
# y_hat= model.predict_generator(test_generator(BS=1))
# y_pred = (y_hat > 0.5)*1
# y_true=test['irrelevant_infer']

# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_true,y_pred))
