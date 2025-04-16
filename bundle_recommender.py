import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations
import random

# Import necessary surprise components
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split # Keep for potential evaluation, though we train on full data here

# --- 1. Data Preparation ---

def prepare_data_for_surprise_and_cooc(transactions_file, categories_file, category_level='Item Category'):
    """
    Loads data, merges categories, prepares data for Surprise, and identifies baskets.

    Args:
        transactions_file (str): Path to transactions.csv
        categories_file (str): Path to product_categories.csv
        category_level (str): Column name from categories_file to use as items.

    Returns:
        tuple: (
            pd.DataFrame: DataFrame suitable for surprise (user, item, rating=1).
            list: List of lists representing baskets for co-occurrence.
            dict: Mapping from original machine_id to surprise inner user id.
            dict: Mapping from surprise inner user id to original machine_id.
            dict: Mapping from original category name to surprise inner item id.
            dict: Mapping from surprise inner item id to original category name.
            pd.DataFrame: The original merged transaction data (useful for lookups).
        ) or None if error occurs
    """
    try:
        df_trans = pd.read_csv(transactions_file)
        df_cats = pd.read_csv(categories_file)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

    # --- Data Validation (Similar to Approach 3) ---
    required_trans_cols = ['machine_id', 'site_session_id', 'prod_category_id']
    required_cat_cols = ['Product Category ID', category_level]
    if not all(col in df_trans.columns for col in required_trans_cols):
        print(f"Transactions file missing required columns: {required_trans_cols}")
        return None
    if 'Product Category ID' not in df_cats.columns and 'prod_category_id' in df_cats.columns:
         df_cats = df_cats.rename(columns={'prod_category_id': 'Product Category ID'})
    elif 'Product Category ID' not in df_cats.columns:
         print(f"Categories file missing 'Product Category ID' column.")
         return None
    if category_level not in df_cats.columns:
        print(f"Categories file missing specified category_level: {category_level}")
        return None
    df_cats = df_cats.rename(columns={'Product Category ID': 'prod_category_id'})
    # --- End Validation ---

    df_merged = pd.merge(
        df_trans,
        df_cats[['prod_category_id', category_level]],
        on='prod_category_id',
        how='left'
    )
    df_merged.dropna(subset=[category_level], inplace=True)
    df_merged[category_level] = df_merged[category_level].astype(str)
    print(f"Loaded {len(df_merged)} transactions with category '{category_level}'.")

    # Prepare data for Surprise (user, item, rating)
    # Use machine_id as user, category_level as item. Implicit feedback -> rating = 1
    df_surprise = df_merged[['machine_id', category_level]].copy()
    df_surprise['rating'] = 1
    # Aggregate: if a user bought the same category multiple times, count as one interaction for model training
    df_surprise = df_surprise.drop_duplicates()

    # Load data into Surprise Dataset
    reader = Reader(rating_scale=(1, 1)) # Rating is always 1 for implicit
    try:
        data = Dataset.load_from_df(df_surprise[['machine_id', category_level, 'rating']], reader)
    except ValueError as e:
        print(f"Error loading data into Surprise: {e}")
        print("Check if 'rating' column has values outside the scale (should be 1).")
        return None


    # Build mappings AFTER building trainset
    trainset = data.build_full_trainset()
    user_map = {trainset.to_raw_uid(inner_id): inner_id for inner_id in trainset.all_users()}
    user_map_reverse = {inner_id: trainset.to_raw_uid(inner_id) for inner_id in trainset.all_users()}
    item_map = {trainset.to_raw_iid(inner_id): inner_id for inner_id in trainset.all_items()}
    item_map_reverse = {inner_id: trainset.to_raw_iid(inner_id) for inner_id in trainset.all_items()}
    print(f"Built Surprise trainset with {trainset.n_users} users and {trainset.n_items} items.")


    # Identify baskets for co-occurrence (same logic as Approach 3)
    baskets_df = df_merged.groupby(['machine_id', 'site_session_id'])[category_level].apply(lambda x: list(set(x))).reset_index()
    baskets_list = [basket for basket in baskets_df[category_level] if len(basket) > 1]
    print(f"Identified {len(baskets_list)} baskets with 2 or more unique items for co-occurrence.")

    return df_surprise, baskets_list, user_map, user_map_reverse, item_map, item_map_reverse, trainset, df_merged

# --- 2. Train Recommendation Model ---

def train_svd_model(trainset):
    """Trains an SVD model on the provided Surprise trainset."""
    print("Training SVD model...")
    algo = SVD(n_factors=100, n_epochs=20, biased=True, random_state=42, verbose=True) # Parameters can be tuned
    algo.fit(trainset)
    print("SVD model training complete.")
    return algo

# --- 3. Calculate Co-occurrence ---

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
                cooc_matrix[item_b][item_a] += 1 # Ensure symmetry
    print(f"Calculated co-occurrences for {len(cooc_matrix)} items.")
    return cooc_matrix

# --- 4. Recommendation Logic ---

def recommend_bundle_approach2(
    user_id, seed_product, model, trainset, cooc_matrix,
    user_map, item_map, item_map_reverse,
    top_n_candidates=50, # How many single recommendations to consider initially
    w_recs = 0.3,       # Weight for individual recommendation scores
    w_cooc_pair = 0.4,  # Weight for co-occurrence between the pair items
    w_cooc_seed = 0.3   # Weight for co-occurrence with the seed item
    ):
    """
    Recommends a 2-product bundle using SVD + Co-occurrence Heuristics.

    Args:
        user_id: The raw ID of the user.
        seed_product: The raw name of the seed product category.
        model: Trained Surprise algorithm (e.g., SVD).
        trainset: Full Surprise trainset.
        cooc_matrix: Pre-calculated co-occurrence counts.
        user_map, item_map, item_map_reverse: Mappings.
        top_n_candidates: Number of initial recommendations to generate.
        w_recs, w_cooc_pair, w_cooc_seed: Weights for the scoring heuristic.

    Returns:
        list: A list containing the names of the two recommended product categories,
              or fewer/empty if not enough recommendations.
    """
    # Convert raw IDs to inner IDs used by Surprise
    if user_id not in user_map:
        print(f"Error: User ID '{user_id}' not found in training data.")
        return []
    if seed_product not in item_map:
        print(f"Error: Seed product '{seed_product}' not found in training data.")
        return []

    inner_user_id = user_map[user_id]
    inner_seed_id = item_map[seed_product]

    print(f"Generating recommendations for user '{user_id}' (Inner ID: {inner_user_id})")
    print(f"Seed product: '{seed_product}' (Inner ID: {inner_seed_id})")

    # 1. Generate Top-N Single Item Recommendations
    all_item_inner_ids = list(item_map_reverse.keys())
    predictions = []
    for inner_item_id in all_item_inner_ids:
        # Don't recommend the seed product itself
        if inner_item_id == inner_seed_id:
            continue
        pred = model.predict(uid=inner_user_id, iid=inner_item_id)
        predictions.append((inner_item_id, pred.est)) # Store (inner_id, predicted_score)

    # Sort by predicted score and get top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n_items = predictions[:top_n_candidates]

    if len(top_n_items) < 2:
        print("Not enough initial recommendations generated to form a pair.")
        return []

    print(f"Generated {len(top_n_items)} initial candidate recommendations.")

    # 2. Generate Candidate Pairs and Score Them
    candidate_pairs = list(combinations(top_n_items, 2))
    scored_pairs = []

    print(f"Scoring {len(candidate_pairs)} candidate pairs...")
    for (item_i_info, item_j_info) in candidate_pairs:
        inner_item_i_id, score_i = item_i_info
        inner_item_j_id, score_j = item_j_info

        # Map back to raw names for co-occurrence lookup
        item_i_name = item_map_reverse.get(inner_item_i_id)
        item_j_name = item_map_reverse.get(inner_item_j_id)

        if not item_i_name or not item_j_name:
            continue # Skip if mapping failed (shouldn't happen)

        # Get co-occurrence scores (handle cases where pairs never co-occurred -> score 0)
        cooc_ij = cooc_matrix.get(item_i_name, {}).get(item_j_name, 0)
        cooc_seed_i = cooc_matrix.get(seed_product, {}).get(item_i_name, 0)
        cooc_seed_j = cooc_matrix.get(seed_product, {}).get(item_j_name, 0)

        # --- Heuristic Scoring Formula ---
        # Normalize scores slightly? For now, use raw values. Weights are key.
        # Consider log-transforming co-occurrence counts if they vary wildly.
        final_score = (w_recs * (score_i + score_j) / 2 +  # Avg recommendation score
                       w_cooc_pair * cooc_ij +             # Co-occurrence of the pair
                       w_cooc_seed * (cooc_seed_i + cooc_seed_j) / 2) # Avg co-occurrence with seed

        scored_pairs.append(((item_i_name, item_j_name), final_score))

    # 3. Rank Pairs and Return Top One
    if not scored_pairs:
        print("No valid pairs could be scored.")
        return []

    scored_pairs.sort(key=lambda x: x[1], reverse=True)

    print(f"Top 5 scored pairs: {scored_pairs[:5]}")

    best_pair = list(scored_pairs[0][0])
    return best_pair


# --- Main Execution Example ---

if __name__ == "__main__":
    TRANSACTIONS_FILE = 'data/transactions.csv' # Replace with your file path
    CATEGORIES_FILE = 'data/product_categories.csv' # Replace with your file path
    CATEGORY_LEVEL = 'Item Sub-Category'       # Or 'Item Sub-Category'

    # 1. Prepare Data
    prep_result = prepare_data_for_surprise_and_cooc(TRANSACTIONS_FILE, CATEGORIES_FILE, CATEGORY_LEVEL)

    if prep_result:
        df_surprise_data, baskets, u_map, u_map_rev, i_map, i_map_rev, ts, df_merged_orig = prep_result

        # 2. Train Model
        svd_model = train_svd_model(ts)

        # 3. Calculate Co-occurrence
        cooc = calculate_cooccurrence(baskets)

        # 4. Get Recommendation
        # --- Replace with actual values ---
        if not u_map or not i_map:
            print("User or Item map is empty, cannot select sample user/item.")
        else:
            sample_user_id = list(u_map.keys())[0] # Example: first user ID
            sample_seed_product = list(i_map.keys())[5] # Example: sixth known item category

            print(f"\n--- Recommending for User: {sample_user_id}, Seed Product: {sample_seed_product} ---")

            # --- Tune Weights Here ---
            recommended_bundle = recommend_bundle_approach2(
                user_id=sample_user_id,
                seed_product=sample_seed_product,
                model=svd_model,
                trainset=ts,
                cooc_matrix=cooc,
                user_map=u_map,
                item_map=i_map,
                item_map_reverse=i_map_rev,
                top_n_candidates=50, # How many initial recommendations
                w_recs = 0.3,        # Weight for SVD scores
                w_cooc_pair = 0.4,   # Weight for pair co-occurrence
                w_cooc_seed = 0.3    # Weight for seed co-occurrence
            )

            if recommended_bundle:
                print(f"\nFinal Recommended Bundle for '{sample_seed_product}': {recommended_bundle}")
            else:
                print("\nCould not generate a bundle recommendation.")
    else:
        print("Data preparation failed. Exiting.")