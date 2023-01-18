import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





def get_summary_statistics_by_country(df):
    # create an empty list to store the results
    results = []
    # get list of all country columns
    country_columns = [col for col in df.columns if col.startswith('is')]
    # loop through each country column
    for column in country_columns:
        # get the data for the current country
        country_data = df[df[column] == 1]
        # get proportion of collusive bids
        count = country_data['labels'].sum()
        proportion = count/len(country_data)
        # get average number of bids per auction, factored out by whether the auction was collusive or not
        avg_num_bids = country_data.groupby('labels').agg({'num_bids':'mean'}).reset_index()
        #get average number of bids per auction for collusive auctions
        collusive_bids = avg_num_bids[avg_num_bids.labels==1]['num_bids'].values[0]
        #get average number of bids per auction for non collusive auctions
        non_collusive_bids = avg_num_bids[avg_num_bids.labels==0]['num_bids'].values[0]
        #get average number of bids per auction
        avg_bids = country_data.num_bids.mean()

        observations = len(country_data)
        # create a dictionary to store the results
        res = {'country': column.replace("is",""), 'number_auctions':observations, 'number_collusive_auctions': count, 'proportion': proportion, 'avg_num_bids': avg_bids, 'avg_num_colbids': collusive_bids, 'avg_num_compbids': non_collusive_bids}
        # append the dictionary to the results list
        results.append(res)
    
    # convert the list to a DataFrame
    results_df = pd.DataFrame(results)
    #calculate the summary statistics for the entire dataset
    total_count = df['labels'].sum()
    total_prop = total_count / len(df)
    total_avg = df.groupby('labels').agg({'num_bids':'mean'}).reset_index()
    collusive_bids_tot = total_avg[total_avg.labels==1]['num_bids'].values[0]
    non_collusive_bids_tot = total_avg[total_avg.labels==0]['num_bids'].values[0]
    total_avg_bids = df.num_bids.mean()

    total_obs = len(df)
    #create a dictionary to store the results
    total_res = {'country': 'total', 'number_auctions': total_obs, 'number_collusive_auctions': total_count, 'proportion': total_prop, 'avg_num_bids': total_avg_bids, 'avg_num_colbids': collusive_bids_tot, 'avg_num_compbids': non_collusive_bids_tot}
    #append the dictionary to the results dataframe
    results_df = results_df.append(total_res, ignore_index=True)
    return results_df


collusion = pd.read_csv("test.csv")


results = get_summary_statistics_by_country(collusion)
print(results)


collusion1 = pd.read_csv("DB_Collusion_All_processed.csv")


def get_bid_margin(df):
    # create an empty list to store the results
    results = []
    # group the data by country
    grouped_data = df.groupby('Dataset')
    # loop through each country group
    for name, group in grouped_data:
        # group the data by auction
        auction_data = group.groupby('Tender')
        # sort the bids in each group by value
        sorted_data = auction_data.Bid_value.apply(lambda x: x.sort_values().reset_index(drop=True))
        # calculate the margin between the lowest and second lowest bid
        margin1 = sorted_data.shift(-1) - sorted_data
        # ensure that the margin is always positive
        margin1 = margin1.apply(lambda x: x if x > 0 else 0)
        # convert the margin to percentage
        margin1_pct = margin1 / sorted_data * 100
        # calculate the average margin
        avg_margin1 = margin1_pct.mean()
        #create a dictionary to store the results
        res = {'country': name, 'avg_margin1': avg_margin1}

        #calculate the margin between the second lowest and third lowest bid
        margin2 = sorted_data.shift(-2) - sorted_data.shift(-1)
        # only process groups with more than 2 bids
        if margin2.count() > 2:
            # ensure that the margin is always positive
            margin2 = margin2.apply(lambda x: x if x > 0 else 0)
            # convert the margin to percentage
            margin2_pct = margin2 / sorted_data.shift(-1) * 100
            # calculate the average margin
            avg_margin2 = margin2_pct.mean()
            # add the new margin to the dictionary
            res.update({'avg_margin2': avg_margin2})
        #append the dictionary to the results list
        results.append(res)
    # convert the list to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def avg_winning_bid_by_country(df):
    # group the data by country
    grouped_data = df.groupby('Dataset')
    # create an empty list to store the results
    results = []
    # loop through each country group
    for name, group in grouped_data:
        # filter for the winning bids
        winning_bids = group[group['Winner'] == 1]
        # calculate the average winning bid
        avg_winning_bid = winning_bids['Bid_value'].mean()
        # append the results to the list
        results.append({'country': name, 'avg_winning_bid': avg_winning_bid})
    # convert the list to a DataFrame
    results_df = pd.DataFrame(results)
    pd.options.display.float_format = '{:.2f}'.format

    return results_df

margin = get_bid_margin(collusion1)
print(margin)

value_contract = avg_winning_bid_by_country(collusion1)
print(value_contract)