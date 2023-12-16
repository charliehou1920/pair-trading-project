import streamlit as st
from metaflow import Flow
from metaflow import get_metadata, metadata
import pandas as pd

# make sure we point the app to the flow folder
# NOTE: MAKE SURE TO RUN THE FLOW AT LEAST ONE BEFORE THIS ;-)
FLOW_NAME = 'PairTrading' # name of the target class
# Set the metadata provider as the src folder in the project,
# which should contains /.metaflow
metadata('../trading')
# Fetch currently configured metadata provider - check it's local!
print(get_metadata())
    
@st.cache_resource
def get_latest_successful_run(flow_name: str):
    "Gets the latest successfull run."
    for r in Flow(flow_name).runs():

        if r.successful: 
            return r

last_run = get_latest_successful_run(FLOW_NAME)
results = last_run.data.all_results

def main():
    # Title of the website
    st.markdown("<h2 style='color: pink;'>Pair Trading with IT stocks in S&P500</h2>", unsafe_allow_html=True)

    # Image showing the clustering result
    st.image("../image/OPTICS_Clustering.png")

    # Brief Summary of The Project
    st.write("This project implements a pairs trading strategy using machine learning methods. There are two parts for this project. In the first part of the project, we used unsupervised learning techniques like OPTICS clustering and statistical methods such as Engle-Granger test to filter out the qualified pairs of stocks for trading. In the second part, A LSTM model was implemented to generate buy and sell signals for trading. We built a basktesting system to evaluate the returns of our pair trading strategy and compared it with the benchmark of S&P 500 index.")

    # Retrieve each pair of stock from the results
    pairs = [result['pair'] for result in results]

    # Create a dropdown select box
    selected_pair = st.selectbox("Select a stock pair:", pairs)

    # Find the data of the selected stocks
    selected_result = next((item for item in results if item['pair'] == selected_pair), None)

    # If the selected stocks are found
    if selected_result:
        # Convert the results to dataframe
        df = pd.DataFrame.from_dict(selected_result['results'], orient='index', columns=['Value'])
        # Display the table
        st.table(df)

    # Generate the plots of the backtests
    if selected_pair == ('CDNS', 'SNPS'):
        st.image("../image/portfolio_CDNS_SNPS.png")
    if selected_pair == ('ADI', 'MCHP'):
        st.image("../image/portfolio_ADI_MCHP.png")
    if selected_pair == ('ADI', 'MPWR'):
        st.image("../image/portfolio_ADI_MPWR.png")
    if selected_pair == ('AMAT', 'NXPI'):
        st.image("../image/portfolio_AMAT_NXPI.png")
    if selected_pair == ('LRCX', 'NXPI'):
        st.image("../image/portfolio_LRCX_NXPI.png")
    if selected_pair == ('QRVO', 'SWKS'):
        st.image("../image/portfolio_QRVO_SWKS.png")
    if selected_pair == ('CSCO', 'TDY'):
        st.image("../image/portfolio_CSCO_TDY.png")
    if selected_pair == ('NTAP', 'TDY'):
        st.image("../image/portfolio_NTAP_TDY.png")
    if selected_pair == ('WDC', 'ZBRA'):
        st.image("../image/portfolio_WDC_ZBRA.png")
        
        
if __name__ == "__main__":
    main()
