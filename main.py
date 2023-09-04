# Import the packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv
from pathlib import Path
from tqdm import tqdm

# Machine Learning
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import adam
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# Import functions and variables from local files
from api_keys import AV_key, FH_key
from AV_API_call import get_earnings
from Add_features import Stock_data

# Load in the ticker list from CSV (Companies currently listed on the S&P 500, NASDAQ 100 and
# DOW Jones Industrial Average).
ticker_file = open("Ticker List.csv", "r", encoding='utf-8-sig')
SL = list(csv.reader(ticker_file, delimiter=","))[0]
ticker_file.close()

# Length of the ticker list to use
n_ticks = len(SL)

# Pull in the data for the simple list
SN = "Earnings_Data.csv"
DN = "All_Data.csv"
OP = Path.cwd() / DN

if not Path(OP).is_file():

    SP = Path.cwd() / SN
    if not Path(OP).is_file():
        Earnings, SL = get_earnings(AV_key, SL, SN)
    else:
        Earnings = pd.read_csv(SP)

    # Create DataFrames for each ticker
    EDF = []
    TC = 1
    n_ticks = int((Earnings.shape[1] - 1) / 6)
    CNs = Earnings.columns.tolist()
    New_tickers = []
    for i in tqdm(range(n_ticks)):
        tick = CNs[TC].split(' ')[1]
        DDF = Earnings.iloc[0:, [TC+1, TC+2, TC+5]]
        DDF['EG_YoY ' + tick] = 100 * (DDF.iloc[:, 1] - DDF.iloc[:, 1].shift(-4)) / DDF.iloc[:, 1].shift(-4)
        DC = 'reportedDate ' + tick
        DDF[DC] = pd.to_datetime(DDF[DC])
        DDF.set_index(DC, inplace=True)
        DDF.drop(columns=['reportedEPS ' + tick], inplace=True)

        EDF.append(DDF.dropna())
        New_tickers.append(tick)
        TC += 6

    # Sweep through the DFs and identify any training points within the dataset
    AC = 0
    LC = 0
    for ticker in tqdm(New_tickers):


        if EDF[LC].shape[0] > 1:
            D_Data = Stock_data(ticker, EDF[LC], 2.0)

        if AC == 0 and D_Data is not None:
            TD = D_Data
            AC += 1
        elif AC > 0 and D_Data is not None:
            TD = np.vstack((TD, D_Data))
            AC += 1

        LC += 1

    # With the earnings data in-hand, scan through the data to find the
    print('Writing saved training data to CSV')
    All_data = pd.DataFrame(TD, columns=['Ticker', 'Earnings Date', 'Earnings Surprise [%]',
                                         'Earnings Growth, YoY [%]', 'ATRP [%]', 'MWR [%]',
                                         'Volume Ratio [%]', 'Gap [%]', 'End State',
                                         'Max Return [%]', 'Day to Max Return'])

    All_data.to_csv('All_data.csv')

else:

    # Read in the data stored previously
    Dummy_DF = pd.read_csv(OP)

    # Data read in from CSV contains the index column
    TCNs = Dummy_DF.columns
    Dummy_DF.drop(TCNs[0:3], axis=1, inplace=True)
    # Dummy_DF.drop(TCNs[9], axis=1, inplace=True)

    # Replace infinite updated data with nan
    Dummy_DF.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN
    Dummy_DF.dropna(inplace=True)
    # All_data = pd.DataFrame(Norm_, columns = Dummy_DF.columns)
    All_data = Dummy_DF

# ===========================================================================================
# Visualize the features
All_data[0].hist()
plt.show


# Use kfolds and cross-validation for train/test
scaler = MinMaxScaler()
X = scaler.fit_transform(All_data.drop(['End State', 'Max Return [%]', 'Day to Max Return'], axis=1))
y = All_data['End State'].to_numpy()

# baseline model
def create_baseline(IVs, DVs):
    # create model
    model = Sequential()
    model.add(Dense(64, input_shape=(IVs,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(DVs, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Use MSE and ADAM
estimator = KerasClassifier(model=create_baseline(X.shape[1], 1), epochs=5, batch_size=32)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))