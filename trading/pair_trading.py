import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from metaflow import FlowSpec, step, IncludeFile
import os


# Function to fetch S&P 500 tickers and sectors
def get_sp500_tickers_and_sectors():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url, header=0)[0]
    return table[['Symbol', 'GICS Sector']].rename(columns={'Symbol': 'Ticker'})

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    ticker = ticker.replace('.', '-')
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        hist.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
        hist['Return'] = hist['Close'] / hist['Close'].shift(1)
        hist.dropna(subset=['Return'], inplace=True)
        return hist
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
    

# Function to generate SP500 stock tables
def generate_sp500_stock_tables(start_date='2020-01-01', end_date='2023-12-05'):
    sp500_tickers_sectors = get_sp500_tickers_and_sectors()
    sp500_data = {}

    for ticker, sector in sp500_tickers_sectors.itertuples(index=False):
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        stock_data['Sector'] = sector
        sp500_data[ticker] = stock_data

    return sp500_data

# Function to standardize the returns in each stock
def standardize_returns(stock_tables):
    scaler = StandardScaler()

    for ticker, df in stock_tables.items():
        if not df.empty and 'Return' in df.columns:
            # Reshape the data for standardization (needs to be 2D)
            returns = df['Return'].values.reshape(-1, 1)

            # Standardize the 'Return' feature
            standardized_returns = scaler.fit_transform(returns)

            # Replace the original 'Return' column with the standardized values
            df['Return'] = standardized_returns

    return stock_tables

# Get tables by sector
def get_tables_by_sector(stock_tables, target_sector):
    filtered_tables = {}
    for ticker, df in stock_tables.items():
        if not df.empty and 'Sector' in df.columns and df['Sector'].iloc[0] == target_sector:
            filtered_tables[ticker] = df
    return filtered_tables

# Calculate PCA for Stocks
def perform_pca(stock_tables, n_components):
    # 1. Create a dataframe which contains the 'Return' of each stock
    returns_data = pd.DataFrame()

    for ticker, df in stock_tables.items():
        if not df.empty and 'Return' in df.columns:
            returns_data[ticker] = df['Return']

    # 2. Compute Covariance Matrix
    cov_matrix = returns_data.cov()

    # 3. Apply PCA with n_components
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(cov_matrix)

    # 4. Create a dataframe with the ticker name and n_components
    principal_components_df = pd.DataFrame(principal_components, index=cov_matrix.index)

    return principal_components_df


# Apply OPTICS clustering algorithm
def cluster_stocks_with_optics(principal_components_df, min_samples):
    
    optics = OPTICS(min_samples=min_samples)
    labels = optics.fit_predict(principal_components_df)

    # labeling
    principal_components_df['Cluster'] = labels

    # Plot
    if principal_components_df.shape[1] in [3, 4]:
        plt.figure(figsize=(10, 7))

        if principal_components_df.shape[1] == 4:
            ax = plt.axes(projection='3d')
            ax.scatter(principal_components_df.iloc[:, 0], principal_components_df.iloc[:, 1], principal_components_df.iloc[:, 2], c=labels, cmap='viridis', marker='o')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
        else:
            plt.scatter(principal_components_df.iloc[:, 0], principal_components_df.iloc[:, 1], c=labels, cmap='viridis', marker='o')
            plt.xlabel('PC1')
            plt.ylabel('PC2')

        plt.title('OPTICS Clustering of IT Stocks in S&P500')
        # Save the plot
        plt.savefig(f'../image/OPTICS_Clustering.png')
        plt.close()

    return principal_components_df

# Show each ticker in every cluster
def display_tickers_in_clusters(clustered_df):

    unique_clusters = clustered_df['Cluster'].unique()


    for cluster in unique_clusters:

        tickers_in_cluster = clustered_df[clustered_df['Cluster'] == cluster].index.tolist()

        print(f"Cluster {cluster}:")
        print(tickers_in_cluster)
        print("\n")

# Store the cluster to lists
def separate_clusters_to_lists(clustered_df):
    # Get clustering labels except -1
    unique_clusters = [cluster for cluster in clustered_df['Cluster'].unique() if cluster != -1]

    # Initialize a dictionary to store the tickers
    clusters_dict = {}

    # Go through every cluster and add the corresponding tickers to the list
    for cluster in unique_clusters:
        clusters_dict[cluster] = clustered_df[clustered_df['Cluster'] == cluster].index.tolist()

    return clusters_dict


# Calculate hurst exponent for time series
def calculate_hurst_exponent(time_series):
    # transfer Series to Numpy
    cumsum = np.cumsum(time_series - np.mean(time_series)).values  
    time_spans = np.arange(2, len(time_series))
    rs_values = []

    for span in time_spans:
        length = len(cumsum) - len(cumsum) % span
        span_series = cumsum[:length]

        # use reshape
        span_series = span_series.reshape(-1, span)
        span_max = span_series.max(axis=1)
        span_min = span_series.min(axis=1)
        r = span_max - span_min
        s = np.std(span_series, axis=1)
        rs = np.mean(r / s)
        rs_values.append(rs)

    hurst_exponent = np.polyfit(np.log(time_spans), np.log(rs_values), 1)[0]
    return hurst_exponent

# Look for the cointegrated Pairs for clusters
def find_cointegrated_pairs_for_clusters(clustered_lists_dict, it_sector_tables):
    results = {}

    for cluster, tickers in clustered_lists_dict.items():
        pairs = []
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                ticker1, ticker2 = tickers[i], tickers[j]

                if ticker1 in it_sector_tables and ticker2 in it_sector_tables:
                    stock1 = it_sector_tables[ticker1]['Close']
                    stock2 = it_sector_tables[ticker2]['Close']

                    # Engle-Granger test
                    score, p_value, _ = coint(stock1, stock2)
                    if p_value < 0.05:
                        # Calculate the hurst exponent for price spread
                        hurst = calculate_hurst_exponent(stock1 - stock2)
                        if hurst < 0.5:
                            pairs.append((ticker1, ticker2))

        results[cluster] = pairs

    return results

# function to preprocess the data for deep learning
def preprocess_data(price_diff, look_back=60):
    X, y = [], []
    for i in range(look_back, len(price_diff)):
        X.append(price_diff[i-look_back:i])
        y.append(price_diff[i])
    return np.array(X), np.array(y)

# Build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')
    return model

# Train the model and apply it on trading
def train_and_evaluate_model_v1(ticker_pair):
    # collect the data and calculate spread
    stock1 = yf.download(ticker_pair[0], start='2015-01-01', end='2023-12-05')['Close']
    stock2 = yf.download(ticker_pair[1], start='2015-01-01', end='2023-12-05')['Close']
    price_difference = stock1 - stock2

    # Normalize the price difference
    # Slide window size for LSTM is 60 (Look back 60 days)
    look_back = 60
    price_difference_normalized = (price_difference - np.mean(price_difference)) / np.std(price_difference)

    # Split the dataset into training, validation, and test sets
    # Training set 70%
    # Validation set 10%
    # Test set 20%
    train_size = int(len(price_difference_normalized) * 0.7)
    validation_size = int(len(price_difference_normalized) * 0.1)
    test_size = len(price_difference_normalized) - train_size - validation_size

    train_data = price_difference_normalized[:train_size]
    validation_data = price_difference_normalized[train_size:train_size + validation_size]
    test_data = price_difference_normalized[-test_size:]  # Adjusted to take the last 20%

    # Preprocess the data in training/validation/test set
    X_train, y_train = preprocess_data(train_data, look_back)
    X_val, y_val = preprocess_data(validation_data, look_back)
    X_test, y_test = preprocess_data(test_data, look_back)

    # Build and train the model
    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')

    # Generate Trading Signal
    predicted_price_diff = model.predict(X_test)
    actual_price_diff = test_data[look_back:look_back + len(predicted_price_diff)]
    percentage_change = (predicted_price_diff[:-1].flatten() - actual_price_diff[:-1]) / actual_price_diff[:-1]

    upper_threshold = 0.3
    lower_threshold = -0.3
    signals = np.where(percentage_change > upper_threshold, 1, np.where(percentage_change < lower_threshold, -1, 0))

    # Back Testing
    initial_cash = 10000
    asset_holdings = np.zeros_like(signals)
    cash = np.zeros_like(signals)
    cash[0] = initial_cash

    # Adjust the loop to align with the length of the test data
    for t in range(1, len(signals)):
        if t < len(test_data.values) - look_back:
            asset_holdings[t] = asset_holdings[t-1] + signals[t-1]
            cash[t] = cash[t-1] - signals[t-1] * test_data.values[t + look_back - 1]

    portfolio_value = initial_cash + asset_holdings * test_data.values[look_back:-1]

    # Evaluations
    final_return = portfolio_value[-1] - initial_cash
    peak = np.maximum.accumulate(portfolio_value)
    drawdown = (peak - portfolio_value) / peak
    max_drawdown = np.max(drawdown)
    portfolio_returns = np.diff(portfolio_value) / portfolio_value[:-1]
    sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns)

    # Plot and save the plot
    sp500_data = yf.download('SPY', start='2020-01-01', end='2023-12-05')['Close']

    # Rescaled S&P 500 data
    sp500_scaled = sp500_data / sp500_data.iloc[0] * initial_cash

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value, label='Portfolio Value')
    plt.plot(sp500_scaled.values, label='S&P 500', alpha=0.7)
    plt.title(f'Portfolio Value Over Time for {ticker_pair[0]} and {ticker_pair[1]} VS S&P500')
    plt.xlabel('Time')
    plt.ylabel('Value in $')
    plt.legend()

    # Make sure the path of 'data' exists
    # if not os.path.exists('data'):
    #     os.makedirs('data')
    plt.savefig(f'../image/portfolio_{ticker_pair[0]}_{ticker_pair[1]}.png')
    plt.close()

    # return the results
    return {
        "Final Return": final_return,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
        "Test Loss": test_loss
    }

# Train the model and apply it on trading(an alternative way, you can skip this part)
def train_and_evaluate_model(ticker_pair):
    # collect the data and calculate spread
    stock1 = yf.download(ticker_pair[0], start='2020-01-01', end='2023-12-05')['Close']
    stock2 = yf.download(ticker_pair[1], start='2020-01-01', end='2023-12-05')['Close']
    price_difference = stock1 - stock2

    # Build slide window and set up the preprocess data
    look_back = 60
    price_difference_normalized = (price_difference - np.mean(price_difference)) / np.std(price_difference)
    X, y = preprocess_data(price_difference_normalized, look_back)

    # Build the model
    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=100, batch_size=32)

    # Genereate Trading Signal
    predicted_price_diff = model.predict(X)
    actual_price_diff = price_difference_normalized[look_back:look_back+len(predicted_price_diff)]
    percentage_change = (predicted_price_diff[:-1].flatten() - actual_price_diff[:-1]) / actual_price_diff[:-1]

    upper_threshold = 0.3
    lower_threshold = -0.3
    signals = np.where(percentage_change > upper_threshold, 1, np.where(percentage_change < lower_threshold, -1, 0))

    # Back Tesing
    initial_cash = 10000
    asset_holdings = np.zeros_like(signals)
    cash = np.zeros_like(signals)
    cash[0] = initial_cash

    for t in range(1, len(signals)):
        asset_holdings[t] = asset_holdings[t-1] + signals[t-1]
        cash[t] = cash[t-1] - signals[t-1] * price_difference.values[look_back+t]

    portfolio_value = initial_cash + asset_holdings * price_difference.values[look_back:-1]

    # Evaluations
    final_return = portfolio_value[-1] - initial_cash
    peak = np.maximum.accumulate(portfolio_value)
    drawdown = (peak - portfolio_value) / peak
    max_drawdown = np.max(drawdown)
    portfolio_returns = np.diff(portfolio_value) / portfolio_value[:-1]
    sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns)

    # Plot and save the plot
    sp500_data = yf.download('SPY', start='2020-01-01', end='2023-12-05')['Close']

    # Rescaled S&P 500 data
    sp500_scaled = sp500_data / sp500_data.iloc[0] * initial_cash

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value, label='Portfolio Value')
    plt.plot(sp500_scaled.values, label='S&P 500', alpha=0.7)
    plt.title(f'Portfolio Value Over Time for {ticker_pair[0]} and {ticker_pair[1]} VS S&P500')
    plt.xlabel('Time')
    plt.ylabel('Value in $')
    plt.legend()

    # Make sure the path of 'data' exists
    # if not os.path.exists('image'):
    #     os.makedirs('data')
    plt.savefig(f'../image/portfolio_{ticker_pair[0]}_{ticker_pair[1]}.png')
    plt.close()

    # return the results
    return {
        "Final Return": final_return,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio
    }


class PairTrading(FlowSpec):
    @step
    def start(self):
        '''
        Step 1: Fetch the S&P 500 stocks from 2020-01-01 to 2023-12-05. Calculate everyday return for each stock
        and standardize them. Obtain the dataframe of stocks which sector is "Information Technology".
        '''

        #Generate the dataframe for each stock in S&P 500
        sp500_stock_tables = generate_sp500_stock_tables()

        # Standardize the returns in each stock
        sp500_stock_tables_standardized = standardize_returns(sp500_stock_tables)

        # Get the dataframe of stocks with 'Information Technology' Sector
        self.it_sector_tables = get_tables_by_sector(sp500_stock_tables_standardized, 'Information Technology')

        self.next(self.clustering)

    @step
    def clustering(self):
        '''
        Step 2: Apply PCA to the dataframe of stocks in IT sector. Set the n_components to 3.
        Then, clustering the data with OPTICS and set the min_samples to 5. Save different clusters of stock
        in a list.
        '''
        # Set n_components to 3 and pass in the stock dataframes in IT sector
        n_components = 3
        principal_components_df = perform_pca(self.it_sector_tables, n_components)

        min_samples = 5  # number of points in a neighborhood for a point to be considered as a core point
        clustered_df = cluster_stocks_with_optics(principal_components_df, min_samples)
        display_tickers_in_clusters(clustered_df)

        # Build the cluster dictionary
        self.clustered_lists_dict = separate_clusters_to_lists(clustered_df)

        self.next(self.pairing)

    @step
    def pairing(self):
        '''
        Step 3: Find the cointegrated pairs of stock in each cluster with Engle-Granger test. Check the 
        stationary of the spread of stocks which have passed Engle-Granger test with hurst exponent.
        '''
        self.cointegrated_pairs_clusters = find_cointegrated_pairs_for_clusters(self.clustered_lists_dict, self.it_sector_tables)
        for cluster, pairs in self.cointegrated_pairs_clusters.items():
            print(f"Cluster {cluster} Pairs of stocks that are cointegrated and whose price differences exhibit mean-reverting behavior: {pairs}")

        self.next(self.trading_with_lstm)

    @step
    def trading_with_lstm(self):
        '''
        Step 4: After obtain the the pairs of stock, use LSTM generate buy and sell signals for trading.
        Apply the LSTM on each pair of the selected stock, run the strategy based on the buy/sell/hold
        signals. Backtest, plot and print the results at last.
        '''
        # Save every pair of stock
        self.all_results = []

        # Go through every pair of stocks in cointegrated_pairs_clusters
        for cluster, pairs in self.cointegrated_pairs_clusters.items():
            for ticker_pair in pairs:
                # apply train_and_evaluate_model function on every pair of stocks
                result = train_and_evaluate_model(ticker_pair)
                # add the result to the list
                self.all_results.append({'pair': ticker_pair, 'results': result})

        # Show the results
        for item in self.all_results:
            print(f"Results for Ticker Pair {item['pair']}:")
            print(f"Final Return: {item['results']['Final Return']}")
            print(f"Max Drawdown: {item['results']['Max Drawdown']}")
            print(f"Sharpe Ratio: {item['results']['Sharpe Ratio']}")
            print("\n")

        self.next(self.end)

    @step
    def end(self):
        print("You have finished the Pair Trading!")       



if __name__ == "__main__":
    PairTrading()




