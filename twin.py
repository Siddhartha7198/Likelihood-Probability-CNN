import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import Model, models
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Lambda, BatchNormalization, Dot
from tensorflow.keras import optimizers	, initializers, regularizers 
from keras.callbacks import EarlyStopping, ModelCheckpoint           
#from tensorflow.keras.utils import plot_model
#import tensorflow_graphics as tfg

import numpy as np
from numpy import *
import math
import random
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split ## to split dataset


inp_data = np.load('img_pair_9m.npy')
out_data = np.load('dis_9m.npy')



def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
    
'''          
def cosine_dis(vects):
	a,b = vects
	ang = tf.math.acos(Dot(axes=1)([a,b])/tf.sqrt(Dot(axes=1)([a,a])*Dot(axes=1)([b,b])))
	return ang
'''

train_in,test_in,train_out,test_out = train_test_split(inp_data,out_data,test_size=0.001,random_state=42)

np.save('test_inp_9md',test_in)
np.save('test_out_9md',test_out)

train_1 = train_in[:,0]
train_2 = train_in[:,1]



#tf.random.set_seed(246)	

tf.random.set_seed(123)
	
input = Input((100, 100, 1))
x = BatchNormalization()(input)
x = Conv2D(32, (2, 2), activation="elu",kernel_initializer='he_uniform')(x)
x = MaxPooling2D(pool_size=(5, 5))(x)
x = Conv2D(64, (2, 2), activation="elu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (2, 2), activation="elu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)

#x = BatchNormalization()(x)
x = Dense(64, activation="sigmoid")(x)
embedding_network = Model(input, x)


input_1 = Input((100, 100, 1))
input_2 = Input((100, 100, 1))


tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = Lambda(euclidean_distance)([tower_1, tower_2])
#normal_layer = BatchNormalization()(merge_layer)
output_layer = Dense(1, activation="linear")(merge_layer)
siamese = Model(inputs=[input_1, input_2], outputs=output_layer)


#siamese = models.load_model('twin_model_21m.hdf5')

siamese.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=0.0001,clipvalue=0.5), metrics=["accuracy"])

earlyStopping = EarlyStopping(monitor='val_loss', patience=12000, verbose=0, mode='min')
mcp_save = ModelCheckpoint('twin_model_9md.hdf5', save_best_only=True, monitor='val_loss', mode='min')


history = siamese.fit([train_1,train_2],train_out,batch_size=128,epochs=12000,callbacks=[mcp_save,earlyStopping],validation_split=0.2,verbose=2,shuffle=True)


np.save('History_9md.npy',history.history)	


