# pair-trading-project
### Setup and Links

<p> - The dependencies of this project are stored in the pyproject.toml which you can find it in the repo. Please use rye commands such as "rye sync" to setup this project. 
<p> - Pair_Trading_Research.ipynb stored in "trading" folder is the research branch for this project. It contains EDA, hyper-parameters tuning, and model selection. We recommend you to run it on Google Colab.
<p> - pair_trading.py stored in "trading" folder this is the dev branch of this project. <b>Run this file first</b> to generate the metaflow data.
<p> - After the metaflow data are generated, run app.py in "app" folder to generate a web application using streamlit.
<p> - The presentation PPT is stored in "presentation" folder.

<p> The link of our results on comet: https://www.comet.com/charlie-hou/pair-trading/view/new/panels

<p> The link of our presentation on Google Slides: https://docs.google.com/presentation/d/1aSGt7mE4wnzOKXTwC3dksQBrlEw2czEXxFZdpfse0ec/edit?usp=sharing

### Introduction

<p>This project implements a pairs trading strategy using machine learning methods. There are two parts for this project. In the first part of the project, we used unsupervised learning techniques like OPTICS clustering and statistical methods such as Engle-Granger test to filter out the qualified pairs of stocks for trading. In the second part, A LSTM model was implemented to generate buy and sell signals for trading. We built a basktesting system to evaluate the returns of our pair trading strategy and compared it with the benchmark of S&P 500 index. The followings are the details of this project:
<p> (1) Dataset: The tickers of 500 stocks from S&P500 were fetched from Wikipedia. The data of stocks were retrieved from Yahoo Finance API.
<p> (2) EDA: We fetched the data, dropped the null values, and standardized the data in our EDA.
<p> (3) Data Preparation: We implemented data realignment and augmentation in our dataset.
<p> (4) Code Structure: There are two branches for our project: research branch and development branch. In our research branch, we run multiple experiments on Google Colab to fine tune the hyper-parameters and select the best model. We also log the results and metrics of the experiments on Comet-ml. In our dev branch, we build a machine learning pipeline with Metaflow. The parameters generated by Metaflow are passed to Streamlit to create an end-to-end user interactive web application.
<p> (5) Training and Optimization: for our data, 70% are set to training set, 10% are set to validation set and 20% are set to test set. We use Adam Optimizer and set learning rate to 0.001.
<p> (6) Tracking: The metrics generated in each epoch, the evaluations for our test set and the results of our backtests are all recorded in Comet.
<p> (7) Testing: We used Mean Square Error as the metric for our test set. We set Final Return, Sharpe Ratio and Max Drawdown as the metrics to evaluate our strategy.
<p> (8) Deployment: We build an end-to-end web application with Streamlit and deploy it on our localhost.
