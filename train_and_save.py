# train_and_save.py
import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations
import joblib

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# --- Functions (Keep prepare_data, train_svd_model, calculate_cooccurrence) ---
# Assume prepare_data_for_surprise_and_cooc, train_svd_model, calculate_cooccurrence are defined above
# Make sure prepare_data_for_surprise_and_cooc returns df_merged_all now

def prepare_data_for_surprise_and_cooc(transactions_file, categories_file, sessions_file, category_level='Item Category'):
    """
    MODIFIED: Loads data, merges categories AND sessions, prepares data for Surprise,
    identifies baskets, AND returns df_merged_all for product name mapping.
    """
    try:
        df_trans = pd.read_csv(transactions_file)
        df_cats = pd.read_csv(categories_file)
        df_sessions = pd.read_csv(sessions_file)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

    # --- Data Validation ---
    required_trans_cols = ['machine_id', 'site_session_id', 'prod_category_id', 'prod_name'] # Added prod_name
    required_cat_cols = ['Product Category ID', category_level]
    required_session_cols = ['machine_id', 'site_session_id', 'household_income', 'children']

    if not all(col in df_trans.columns for col in required_trans_cols): print(f"Transactions missing cols"); return None
    if 'Product Category ID' not in df_cats.columns and 'prod_category_id' in df_cats.columns: df_cats = df_cats.rename(columns={'prod_category_id': 'Product Category ID'})
    elif 'Product Category ID' not in df_cats.columns: print(f"Categories missing 'Product Category ID'"); return None
    if category_level not in df_cats.columns: print(f"Categories missing '{category_level}'"); return None
    if not all(col in df_sessions.columns for col in required_session_cols): print(f"Sessions missing cols: {required_session_cols}"); return None

    df_cats = df_cats.rename(columns={'Product Category ID': 'prod_category_id'})
    # --- End Validation ---

    # Merge transactions and categories
    df_merged_cat = pd.merge(
        df_trans,
        df_cats[['prod_category_id', category_level]],
        on='prod_category_id',
        how='left'
    )
    df_merged_cat.dropna(subset=[category_level, 'prod_name'], inplace=True) # Also drop if prod_name is missing
    df_merged_cat[category_level] = df_merged_cat[category_level].astype(str)
    df_merged_cat['prod_name'] = df_merged_cat['prod_name'].astype(str)

    # Merge session data
    df_sessions_unique = df_sessions[['machine_id', 'site_session_id', 'household_income', 'children']].drop_duplicates(subset=['machine_id', 'site_session_id'])
    df_merged_all = pd.merge(
        df_merged_cat,
        df_sessions_unique,
        on=['machine_id', 'site_session_id'],
        how='left'
    )
    df_merged_all['household_income'].fillna(99, inplace=True)
    df_merged_all['children'].fillna(99, inplace=True)

    print(f"Loaded {len(df_merged_all)} transactions with category '{category_level}' and session info.")

    # --- Prepare for Surprise ---
    df_surprise = df_merged_all[['machine_id', category_level]].copy()
    df_surprise['rating'] = 1
    df_surprise = df_surprise.drop_duplicates()
    reader = Reader(rating_scale=(1, 1))
    try:
        data = Dataset.load_from_df(df_surprise[['machine_id', category_level, 'rating']], reader)
    except ValueError as e:
        print(f"Error loading data into Surprise: {e}")
        return None

    trainset = data.build_full_trainset()
    user_map = {trainset.to_raw_uid(inner_id): inner_id for inner_id in trainset.all_users()}
    user_map_reverse = {inner_id: trainset.to_raw_uid(inner_id) for inner_id in trainset.all_users()}
    item_map = {trainset.to_raw_iid(inner_id): inner_id for inner_id in trainset.all_items()}
    item_map_reverse = {inner_id: trainset.to_raw_iid(inner_id) for inner_id in trainset.all_items()}
    print(f"Built Surprise trainset with {trainset.n_users} users and {trainset.n_items} items.")

    # --- Identify Baskets ---
    baskets_df = df_merged_all.groupby(['machine_id', 'site_session_id'])[category_level].apply(lambda x: list(set(x))).reset_index()
    baskets_list = [basket for basket in baskets_df[category_level] if len(basket) > 1]
    print(f"Identified {len(baskets_list)} baskets with 2+ unique items for co-occurrence.")

    # --- Create Category -> Product Names Map ---
    print("Creating category to product names map...")
    category_to_products_map = df_merged_all.groupby(category_level)['prod_name'].apply(lambda x: list(x.unique())).to_dict()
    print(f"Created map for {len(category_to_products_map)} categories.")

    # Select necessary columns for user lookup dataframe
    df_user_lookup = df_merged_all[['machine_id', 'household_income', 'children']].drop_duplicates().reset_index(drop=True)

    # Return the map along with other items
    return (df_surprise, baskets_list, user_map, user_map_reverse, item_map, item_map_reverse,
            trainset, df_user_lookup, list(item_map.keys()), category_to_products_map) # Added map

# --- Functions train_svd_model, calculate_cooccurrence (Keep as before, ensure calculate_cooccurrence returns dict of dicts) ---
# ... make sure calculate_cooccurrence saves dict of dicts ...
def calculate_cooccurrence(baskets_list):
    """Calculates co-occurrence counts between items based on baskets."""
    cooc_matrix = defaultdict(Counter)
    print("Calculating co-occurrence matrix...")
    for basket in baskets_list:
        unique_items = sorted(list(set(basket)))
        for i in range(len(unique_items)):
            for j in range(i + 1, len(unique_items)):
                item_a = unique_items[i]
                item_b = unique_items[j]
                cooc_matrix[item_a][item_b] += 1
                cooc_matrix[item_b][item_a] += 1
    # Convert defaultdict of Counters to dict of dicts for saving
    cooc_dict = {k: dict(v) for k, v in cooc_matrix.items()}
    print(f"Calculated co-occurrences for {len(cooc_dict)} items.")
    return cooc_dict

def train_svd_model(trainset):
    """Trains an SVD model on the provided Surprise trainset."""
    print("Training SVD model...")
    algo = SVD(n_factors=100, n_epochs=20, biased=True, random_state=42, verbose=False) # Verbose=False for cleaner saving script output
    algo.fit(trainset)
    print("SVD model training complete.")
    return algo

# --- Main Saving Execution ---
if __name__ == "__main__":
    TRANSACTIONS_FILE = 'data/transactions.csv'
    CATEGORIES_FILE = 'data/product_categories.csv'
    SESSIONS_FILE = 'data/sessions.csv'
    CATEGORY_LEVEL = 'Item Category'

    print("Starting data preparation and model training...")
    # Capture the returned map
    prep_result = prepare_data_for_surprise_and_cooc(TRANSACTIONS_FILE, CATEGORIES_FILE, SESSIONS_FILE, CATEGORY_LEVEL)

    if prep_result:
        # Unpack including the new map
        df_surprise_data, baskets, u_map, u_map_rev, i_map, i_map_rev, ts, df_users, product_list, cat_prod_map = prep_result

        # Train Model
        svd_model = train_svd_model(ts)

        # Calculate Co-occurrence
        cooc = calculate_cooccurrence(baskets)

        # Save components
        print("Saving components...")
        joblib.dump(svd_model, 'svd_model.joblib')
        joblib.dump(cooc, 'cooccurrence_matrix.joblib')
        joblib.dump(u_map, 'user_map.joblib')
        joblib.dump(i_map, 'item_map.joblib')
        joblib.dump(i_map_rev, 'item_map_reverse.joblib')
        joblib.dump(df_users, 'user_lookup_data.joblib')
        joblib.dump(product_list, 'product_list.joblib')
        joblib.dump(cat_prod_map, 'category_to_products_map.joblib') # Save the new map

        print("All components saved successfully.")
    else:
        print("Failed to prepare data or train model. Nothing saved.")