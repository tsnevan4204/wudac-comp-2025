# train_and_save.py - UPDATED with Evaluation and Plots

import pandas as pd
from collections import defaultdict, Counter
import joblib
import random
import time

# Import necessary surprise components
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split # For splitting data

# Import for evaluation metrics
from collections import defaultdict

# Import for plotting
import matplotlib.pyplot as plt
import numpy as np

# --- Functions (prepare_data, calculate_cooccurrence) ---
# Make sure prepare_data returns df_merged_all for cat_prod_map creation
def prepare_data_for_surprise_and_cooc(transactions_file, categories_file, sessions_file, category_level='Item Category'):
    """ Loads data, merges, prepares for Surprise, identifies baskets, returns full merged df """
    try:
        df_trans = pd.read_csv(transactions_file)
        df_cats = pd.read_csv(categories_file)
        df_sessions = pd.read_csv(sessions_file)
    except FileNotFoundError as e: print(f"Error loading data: {e}"); return None
    # --- Data Validation ---
    req_t = ['machine_id', 'site_session_id', 'prod_category_id', 'prod_name']; req_c = ['Product Category ID', category_level]; req_s = ['machine_id', 'site_session_id', 'household_income', 'children']
    if not all(c in df_trans.columns for c in req_t): print(f"Trans missing cols"); return None
    if 'Product Category ID' not in df_cats.columns and 'prod_category_id' in df_cats.columns: df_cats = df_cats.rename(columns={'prod_category_id': 'Product Category ID'})
    elif 'Product Category ID' not in df_cats.columns: print(f"Cats missing 'Product Category ID'"); return None
    if category_level not in df_cats.columns: print(f"Cats missing '{category_level}'"); return None
    if not all(c in df_sessions.columns for c in req_s): print(f"Sessions missing cols: {req_s}"); return None
    df_cats = df_cats.rename(columns={'Product Category ID': 'prod_category_id'})
    # --- End Validation ---
    df_merged_cat = pd.merge(df_trans, df_cats[['prod_category_id', category_level]], on='prod_category_id', how='left')
    df_merged_cat.dropna(subset=[category_level, 'prod_name'], inplace=True); df_merged_cat[category_level] = df_merged_cat[category_level].astype(str); df_merged_cat['prod_name'] = df_merged_cat['prod_name'].astype(str)
    df_sessions_unique = df_sessions[req_s].drop_duplicates(subset=['machine_id', 'site_session_id'])
    df_merged_all = pd.merge(df_merged_cat, df_sessions_unique, on=['machine_id', 'site_session_id'], how='left')
    df_merged_all['household_income'].fillna(99, inplace=True); df_merged_all['children'].fillna(99, inplace=True)
    print(f"Loaded {len(df_merged_all)} transactions with category '{category_level}' and session info.")
    # --- Prepare for Surprise ---
    df_surprise = df_merged_all[['machine_id', category_level]].copy(); df_surprise['rating'] = 1; df_surprise = df_surprise.drop_duplicates()
    reader = Reader(rating_scale=(1, 1));
    try: data = Dataset.load_from_df(df_surprise[['machine_id', category_level, 'rating']], reader)
    except ValueError as e: print(f"Error loading data into Surprise: {e}"); return None
    # --- Identify Baskets ---
    baskets_df = df_merged_all.groupby(['machine_id', 'site_session_id'])[category_level].apply(lambda x: list(set(x))).reset_index()
    baskets_list = [basket for basket in baskets_df[category_level] if len(basket) > 1]
    print(f"Identified {len(baskets_list)} baskets with 2+ unique items.")
    # --- Create Maps and User Lookup (using full dataset before split) ---
    full_trainset_for_maps = data.build_full_trainset() # Use full data for consistent maps
    user_map = {full_trainset_for_maps.to_raw_uid(iid): iid for iid in full_trainset_for_maps.all_users()}
    user_map_reverse = {iid: full_trainset_for_maps.to_raw_uid(iid) for iid in full_trainset_for_maps.all_users()}
    item_map = {full_trainset_for_maps.to_raw_iid(iid): iid for iid in full_trainset_for_maps.all_items()}
    item_map_reverse = {iid: full_trainset_for_maps.to_raw_iid(iid) for iid in full_trainset_for_maps.all_items()}
    category_to_products_map = df_merged_all.groupby(category_level)['prod_name'].apply(lambda x: list(x.unique())).to_dict()
    df_user_lookup = df_merged_all[['machine_id', 'household_income', 'children']].drop_duplicates().reset_index(drop=True)
    product_list = sorted(list(item_map.keys())) # Get product list from item map keys

    return (data, baskets_list, user_map, user_map_reverse, item_map, item_map_reverse,
            df_user_lookup, product_list, category_to_products_map)


def calculate_cooccurrence(baskets_list):
    """ Calculates co-occurrence counts between items based on baskets. """
    cooc_matrix = defaultdict(Counter)
    print("Calculating co-occurrence matrix...")
    for basket in baskets_list:
        unique_items = sorted(list(set(basket)))
        for i in range(len(unique_items)):
            for j in range(i + 1, len(unique_items)):
                item_a, item_b = unique_items[i], unique_items[j]
                cooc_matrix[item_a][item_b] += 1; cooc_matrix[item_b][item_a] += 1
    cooc_dict = {k: dict(v) for k, v in cooc_matrix.items()} # Convert for saving
    print(f"Calculated co-occurrences for {len(cooc_dict)} items.")
    return cooc_dict


# --- !! NEW Evaluation Functions !! ---

def precision_recall_f1_hr_at_k(predictions, k=10, threshold=1):
    """Return average precision, recall, F1 and Hit Rate at k."""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    f1s = dict()
    hit_rates = dict()

    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k]) # Not really needed for est

        # Number of relevant AND recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold)) for (_, true_r) in user_ratings[:k]
        ) # In implicit case, true_r is always 1 if item is in test set for user

        # Precision@k: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / k if k != 0 else 0

        # Recall@k: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        # F1@k
        if precisions[uid] + recalls[uid] != 0:
            f1s[uid] = 2 * (precisions[uid] * recalls[uid]) / (precisions[uid] + recalls[uid])
        else:
            f1s[uid] = 0

        # Hit Rate@k: Did we recommend at least one relevant item?
        hit_rates[uid] = 1 if n_rel_and_rec_k > 0 else 0


    # Average metrics over all users
    avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
    avg_recall = sum(rec for rec in recalls.values()) / len(recalls)
    avg_f1 = sum(f1 for f1 in f1s.values()) / len(f1s)
    avg_hit_rate = sum(hr for hr in hit_rates.values()) / len(hit_rates)


    return avg_precision, avg_recall, avg_f1, avg_hit_rate

# --- Main Execution ---
if __name__ == "__main__":
    TRANSACTIONS_FILE = 'data/transactions.csv'
    CATEGORIES_FILE = 'data/product_categories.csv'
    SESSIONS_FILE = 'data/sessions.csv'
    CATEGORY_LEVEL = 'Item Category'
    K_FOR_EVAL = 10 # Evaluate Top-10 recommendations

    print("Starting data preparation...")
    prep_result = prepare_data_for_surprise_and_cooc(TRANSACTIONS_FILE, CATEGORIES_FILE, SESSIONS_FILE, CATEGORY_LEVEL)

    if prep_result:
        # Unpack results
        surprise_data, baskets, u_map, u_map_rev, i_map, i_map_rev, df_users, prod_list, cat_prod_map = prep_result

        # --- !! Train/Test Split !! ---
        print(f"\nSplitting data (using random_state=42 for reproducibility)...")
        trainset, testset = train_test_split(surprise_data, test_size=0.20, random_state=42)
        print("Split complete.")

        # --- Train SVD Model (on Training set only for evaluation) ---
        print("\nTraining SVD model on TRAINING set for evaluation...")
        start_time = time.time()
        # Use slightly more robust params for potentially smaller trainset
        algo_eval = SVD(n_factors=100, n_epochs=25, biased=True, random_state=42, verbose=False)
        algo_eval.fit(trainset)
        print(f"SVD training for evaluation complete. Time: {time.time() - start_time:.2f}s")

        # --- Evaluate Model ---
        print(f"\nEvaluating model on TEST set (calculating Precision/Recall/F1/HitRate @{K_FOR_EVAL})...")
        start_time = time.time()
        predictions_test = algo_eval.test(testset)
        avg_precision, avg_recall, avg_f1, avg_hit_rate = precision_recall_f1_hr_at_k(predictions_test, k=K_FOR_EVAL)
        print(f"Evaluation complete. Time: {time.time() - start_time:.2f}s")

        print("\n--- Evaluation Results ---")
        print(f"Average Precision@{K_FOR_EVAL}: {avg_precision:.4f}")
        print(f"Average Recall@{K_FOR_EVAL}:    {avg_recall:.4f}")
        print(f"Average F1@{K_FOR_EVAL}:        {avg_f1:.4f}")
        print(f"Average Hit Rate@{K_FOR_EVAL}:  {avg_hit_rate:.4f}")
        print("--------------------------")

        # --- !! Generate and Save Plots !! ---
        print("\nGenerating evaluation plots...")
        metrics_names = [f'Precision@{K_FOR_EVAL}', f'Recall@{K_FOR_EVAL}', f'F1@{K_FOR_EVAL}', f'HitRate@{K_FOR_EVAL}']
        metrics_values = [avg_precision, avg_recall, avg_f1, avg_hit_rate]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.ylabel("Average Score")
        plt.title("Recommendation Model Evaluation Metrics")
        plt.ylim(0, max(metrics_values) * 1.15) # Set y-limit slightly above max value

        # Add value labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center') # Add text label

        plt.tight_layout()
        plot_filename = 'evaluation_metrics_plot.png'
        plt.savefig(plot_filename)
        print(f"Saved evaluation plot to {plot_filename}")
        plt.close() # Close the plot

        # --- !! Retrain Model on FULL Dataset for Saving !! ---
        print("\nRetraining SVD model on FULL dataset for application use...")
        start_time = time.time()
        full_trainset = surprise_data.build_full_trainset()
        algo_final = SVD(n_factors=100, n_epochs=20, biased=True, random_state=42, verbose=False)
        algo_final.fit(full_trainset)
        print(f"SVD retraining on full data complete. Time: {time.time() - start_time:.2f}s")


        # --- Calculate Co-occurrence (using full dataset's baskets) ---
        cooc = calculate_cooccurrence(baskets)

        # --- Save Components (including the FINAL model) ---
        print("\nSaving final components for web app...")
        joblib.dump(algo_final, 'svd_model.joblib') # Save the model trained on FULL data
        joblib.dump(cooc, 'cooccurrence_matrix.joblib')
        joblib.dump(u_map, 'user_map.joblib')
        joblib.dump(i_map, 'item_map.joblib')
        joblib.dump(i_map_rev, 'item_map_reverse.joblib')
        joblib.dump(df_users, 'user_lookup_data.joblib')
        joblib.dump(prod_list, 'product_list.joblib')
        joblib.dump(cat_prod_map, 'category_to_products_map.joblib')

        print("\n--- Process Complete ---")
        print("Saved final SVD model (trained on full data), co-occurrence matrix, data maps, user lookup data, product list, category map, and evaluation plot.")

    else:
        print("Data preparation failed. Exiting.")