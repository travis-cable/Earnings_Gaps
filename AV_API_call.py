# Import packages
from alpha_vantage.timeseries import TimeSeries
from tqdm import tqdm
import pandas as pd
import json
import requests
import time

def get_earnings(key, tickers, output_path = None):

    # Loop through the list of tickers, pulling quarterly earnings data from Alpha Vantage
    LC = 1
    EF = 0
    for ticker in tqdm(tickers):

        # Create a URL for each of the tickers in the style specified by AV
        url = 'https://www.alphavantage.co/query?function=EARNINGS&symbol={}&apikey={}'.format(ticker[0], key)

        # Run the API call
        r = requests.get(url)

        # Convert the API pull to .json and then to DataFrame
        data = r.json()

        # Try to pull the quarterly earnings data from the json file
        try:
            QE = data['quarterlyEarnings']
        except:
            EF = 1
            continue

        # Use a dummy dataframe to load the data into
        DDF = pd.DataFrame(QE)

        # Add the ticker symbol to the column names
        DDF.columns = [col + ' ' + ticker[0] for col in DDF.columns]

        # Join the existing ticker data and then recently acquired
        if LC == 1 and EF == 0:
            EDF = DDF
        elif LC > 1 and EF == 0:
            EDF = pd.concat([EDF, DDF], axis=1)

        # Increment the counter
        LC += 1
        EF = 0

    if output_path == None:
        print('Finished loading in the data, but no filepath provided. Data not written to file.')
        print('')
    else:
        EDF.to_csv(output_path + 'Earnings_data.csv')
        print(f'Finished loading in data. CSV file saved to {output_path}.')

    return EDF