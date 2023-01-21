"""This script takes the original DB_Collusion_All_processed dataset, and adds
a column 'original_Bid_value'. These are obtained from the country-specific
datasets for the auctions.
"""
import numpy as np
import pandas as pd

def original_auction(country_datasets, tender_id, auction_all, country):
    """Takes an auctions Tender number from the DB_Collusion_All_processed,
    and returns a pandas dataframe view of that auction in the original
    country specific dataset."""
    print(tender_id)

    df_country = country_datasets[country]

    identifiers = ['Number_bids','Collusive_competitor','CV','SPD','DIFFP','RD']
    matches = df_country[identifiers] == auction_all.iloc[0][identifiers]
    matches_all = matches.apply(lambda x: np.sum(x)/len(identifiers), axis=1) == 1

    if auction_all.iloc[0]['Number_bids'] != np.sum(matches_all):
        #found more matches than original number of bids in the auction
        matches_all = matches_all.apply(lambda x: False)


    return df_country[matches_all]

filepaths = ["./DB_Collusion_Brazil_processed.csv",
             "./DB_Collusion_Italy_processed.csv",
             "./DB_Collusion_America_processed.csv",
             "./DB_Collusion_Switzerland_GR_and_See-Gaster_processed.csv",
             "./DB_Collusion_Switzerland_Ticino_processed.csv",
             "./DB_Collusion_Japan_processed.csv",
             "./DB_Collusion_All_processed.csv"]

bid_names = ['Bid_value',
             'Bid_value',
             'Bid_value_without_inflation',
             'Bid_value',
             'Bid_value',
             'Bid_value']

data_all_np = np.genfromtxt(filepaths[-1], delimiter=",", skip_header=1)
country_datasets = [pd.read_csv(fp, header=0) for fp in filepaths]
data_all = country_datasets[-1]
data_all['original_Bid_value'] = np.full(data_all_np.shape[0],np.nan)
idx = np.zeros(data_all_np.shape[0], dtype=bool)

for i in np.unique(data_all_np[:, 0]):
    auction_all = data_all[data_all['Tender'] == i].copy()
    sorted_idx = np.argsort(auction_all['Bid_value'])
    country = int(auction_all.iloc[0]['Dataset'])

    og_auction = original_auction(country_datasets, i, auction_all, country)
    og_bids = np.sort(og_auction[bid_names[country]])

    if og_bids.size > 0:
        og_bid_idx = 0
        for idx in sorted_idx:
            auction_all.iloc[idx, -1] = og_bids[og_bid_idx]
            og_bid_idx += 1

        data_all[data_all['Tender'] == i] = auction_all

data_all.to_csv('./DB_Collusion_All_processed2.csv',index=False)
