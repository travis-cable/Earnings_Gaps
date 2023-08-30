# Import the packages
import yfinance as yf
import pandas as pd
import csv

# Import functions and variables from local files
from api_keys import AV_key
from AV_API_call import get_earnings

# Load in the ticker list from CSV (Companies currently listed on the S&P 500, NASDAQ 100 and
# DOW Jones Industrial Average).
ticker_file = open("Ticker List.csv", "r", encoding='utf-8-sig')
SL = list(csv.reader(ticker_file, delimiter=","))
ticker_file.close()

# Pull in the data for the simple list
OP = "Users\\traviscable\\Documents\\Term 3 Project\\Earnings_Gaps\\AV Outputs"
Earnings = get_earnings(AV_key, SL, OP)
