import keras.optimizers
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Machine Learning
import tensorflow as tf
from keras.models import load_model
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Built-ins
from random import choice

class NeuralNetwork():
    def __init__(self):
        self.model = Sequential()

    def build(self, layers, npl, LA, drop_percent, input_shape):

        # Add a certian number of layers
        for i in range(layers):
            if i == 0:
                self.model.add(Dense(npl[i], activation=LA[i], input_dim=input_shape))
            else:
                self.model.add(Dense(npl[i], activation=LA[i]))

            self.model.add(Dropout(rate=drop_percent[i]))

        # Output layer for the binary classification task
        self.model.add(Dense(1, activation='sigmoid'))

    def compile(self, optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=['mean_squared_error'], metrics=['mae']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, x_train, y_train, batch_size, epochs, CBs=None, validation_data=None,
              split = None, verbose=1):
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                 callbacks=CBs, validation_split=split, validation_data=validation_data,
                                 verbose=verbose)
        return history

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)

save_string = 'all_earnings_events'
earnings_file_extension = f'data/{save_string}.parquet'
train_column = 'label_after_1_days'

# Set the random seed for TensorFlow
tf.random.set_seed(42)

# Aggregate news to a single embedding per day, per ticker, by averaging embeddings if
# the same ticker has multiple articles for a single day
earnings_events_df = pd.read_parquet(earnings_file_extension)
earnings_events_df.dropna(inplace=True)
earnings_events_df.sort_index(level='date', inplace=True)
# earnings_events_df = earnings_events_df[~earnings_events_df.isin([np.inf, -np.inf]).any(axis=1)]

column_names = earnings_events_df.columns

# Split the data into dependent and independent variable sets
X = earnings_events_df[column_names[0:21]].to_numpy()

# Try the next day returns as the only dependent variable
dependent_df = earnings_events_df[train_column]

# Classify articles that came out after positive day returns as "1", else "0"
YC = dependent_df.to_numpy().astype(int)

# Split the data into train and test
test_fraction = 0.15
val_fraction = 0
train_fraction = 1.0 - test_fraction - val_fraction

train_row = int(X.shape[0] * train_fraction)
val_row = int(X.shape[0] * (train_fraction + val_fraction))

X_train = X[0:train_row, :]
# X_val = X[(train_row+1):val_row, :]
X_test = X[(val_row + 1):, :]

y_train = YC[0:train_row]
# y_val = YC[(train_row+1):val_row]
y_test = YC[(val_row + 1):]

# Scale the data with a standard scaler
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2 = StandardScaler()
scaled = scaler2.fit(X_train)
scaled_X_train = scaled.transform(X_train).astype(np.float32)
# scaled_X_val = scaled.transform(X_val).astype(np.float32)
scaled_X_test = scaled.transform(X_test).astype(np.float32)

train = 1

if train == 1:

    # Define a list of hyperparameters to test
    dummy = []
    feature_size = scaled_X_train.shape[1]
    HP = {}
    HP['HLs'] = [1, 2, 3, 4]
    HP['NPL'] = [32, 64, 128, 256]
    HP['DR'] = [0.0, 0.10, 0.20]
    HP['batch_size'] = [50, 100, 150]
    HP['AF'] = ['relu', 'tanh']

    # Try 50 different combinations of the hyperparameters and see if we can get something that fits well
    combs = 50
    cc = 1
    SCs = []
    epochs = 100
    MVA = 0

    # Define the EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='accuracy',
        patience=10,
        restore_best_weights=True
    )

    for i in range(combs):

        # Choose the hyperparameters randomly and append the list to a dictionary
        CPs = {}
        CPs['CBS'] = choice(HP['batch_size']) # Batch size
        CPs['CLs'] = choice(HP['HLs'])  # Layers
        CPs['CNPLs'] = [choice(HP['NPL'])]

        for j in range(CPs['CLs'] - 1):
            CPs['CNPLs'].append(CPs['CNPLs'][0] / (2**(j+1)))

        CPs['CDRs'] = [choice(HP['DR']) for k in range(CPs['CLs'])]     # Dropout per layer
        CPs['CAs'] = [choice(HP['AF']) for k in range(CPs['CLs'])]      # Activation function per layer

        attempts = 0
        while CPs in SCs and attempts < 10:
            CPs = {}
            CPs['CBS'] = choice(HP['batch_size'])  # Batch size
            CPs['CLs'] = choice(HP['HLs'])  # Layers
            CPs['CNPLs'] = [choice(HP['NPL'])]

            for j in range(CPs['CLs'] - 1):
                CPs['CNPLs'].append(CPs['CNPLs'][0] / (2 ** (j + 1)))

            CPs['CDRs'] = [choice(HP['DR']) for k in range(CPs['CLs'])]  # Dropout per layer
            CPs['CAs'] = [choice(HP['AF']) for k in range(CPs['CLs'])]  # Activation function per layer
            attempts += 1

        # Couldn't find a new combination to try
        if attempts == 10:
            break

        # Print out the combination and the iteration we're on then store the current combination
        print(f'The best validation accuracy thus far is {MVA}')
        print('')
        print(f'Trying combination {i}: {CPs}')
        print('')
        SCs.append(CPs)

        model = NeuralNetwork()
        model.build(CPs['CLs'], CPs['CNPLs'], CPs['CAs'], CPs['CDRs'], scaled_X_train.shape[1])
        model.compile(loss='binary_crossentropy', metrics=['accuracy'])
        hist = model.train(x_train=scaled_X_train, y_train=y_train, epochs=epochs, batch_size=CPs['CBS'],
                           CBs=[early_stopping])#, validation_data=(scaled_X_val, y_val))

        model_accuracy = np.mean(hist.history['accuracy'][-5:])

        if model_accuracy > MVA:
            Best_params = CPs
            MVA = model_accuracy

        print('=====================================================================')
        print('')

    print(f'The best model is comprised of {Best_params}')
    print('')

    best_model = NeuralNetwork()
    best_model.build(Best_params['CLs'], Best_params['CNPLs'], Best_params['CAs'], Best_params['CDRs'],
                     scaled_X_train.shape[1])
    best_model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    hist = best_model.train(x_train=scaled_X_train, y_train=y_train, epochs=100, batch_size=CPs['CBS'])

    best_model.model.save('data/best_model_'+save_string+'_'+train_column)

else:

    best_model = load_model('data/best_model_'+save_string+'_'+train_column)


# With the best model determined, predict the labels for the dataset
test = best_model.evaluate(scaled_X_test, y_test)

print(f'The {train_column} loss and accuracy are: {np.round(test[0],3)}, {np.round(test[1]*100,1)}')
