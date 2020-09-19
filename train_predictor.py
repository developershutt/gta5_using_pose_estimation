import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import LSTM, Flatten, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.optimizers import Adam

df = pd.read_csv('coordinates_dataset.csv', sep=',')

standing = df[df['LABEL'] == 'STANDING']
walking = df[df['LABEL'] == 'WALKING']
running = df[df['LABEL'] == 'RUNNING']

def split(df):
    return int(df.shape[0] * 0.8)
train_df = pd.DataFrame(columns = df.columns)
test_df = pd.DataFrame(columns = df.columns)

train_df = pd.concat([train_df,standing[:split(standing)]])
train_df = pd.concat([train_df,walking[:split(walking)]])
train_df = pd.concat([train_df,running[:split(running)]])

test_df = pd.concat([test_df,standing[split(standing):]])
test_df = pd.concat([test_df,walking[split(walking):]])
test_df = pd.concat([test_df,running[split(running):]])

#seperate labels from dataset
train_labels = train_df['LABEL']
test_labels = test_df['LABEL']

#drop label from dataset
train_df.drop(columns = ['LABEL'], inplace = True)
test_df.drop(columns = ['LABEL'], inplace = True)

# Convert dataframe to numpy array
X_train = train_df.to_numpy()
X_test = test_df.to_numpy()

# scale dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# encode labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(train_labels.to_numpy())
y_test = encoder.transform(test_labels.to_numpy())

# convert data into time series
n_time_steps = 16
n_features = 34

train_gen = TimeseriesGenerator(X_train, y_train, length = n_time_steps, batch_size = 128)
test_gen = TimeseriesGenerator(X_test, y_test, length = n_time_steps, batch_size = 128)


model = Sequential()
model.add(LSTM(32, return_sequences = True, input_shape = (n_time_steps, n_features), kernel_regularizer = l2(0.000001), bias_regularizer = l2(0.000001),name = 'lstm-1'))
model.add(Flatten(name='flatten'))
model.add(Dense(128, activation='relu', name='dense_1',kernel_regularizer = l2(0.000001), bias_regularizer = l2(0.000001)))
model.add(Dense(len(np.unique(y_train)), activation='softmax', kernel_regularizer = l2(0.000001), bias_regularizer = l2(0.000001), name='output'))          
print(model.summary())

model.compile(loss= 'sparse_categorical_crossentropy', optimizer = Adam(), metrics =['accuracy'])

callbacks = [ModelCheckpoint('_models/model.h5', save_weights_only = False, save_best_only = True, verbose = 1)]

history = model.fit_generator(train_gen, epochs=5, validation_data=test_gen, callbacks=callbacks)

plt.title('Train accuracy')
plt.plot(history.history['acc'])
plt.show()
plt.title('Train loss')
plt.plot(history.history['loss'])
plt.show()

# store objects
with open('objects/scaler.pickle', 'wb') as handle:
    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('objects/label_encoder.pickle', 'wb') as handle:
    pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
