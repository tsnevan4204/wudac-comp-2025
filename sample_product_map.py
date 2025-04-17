# sample_product_map.py

# Using placehold.co for generic images. Replace with real URLs if desired.
# Structure: Category Name -> List of {'name': 'Product Name', 'img': 'image_url'}
SAMPLE_PRODUCT_DATA = {
    "Laptops": [
        {"name": "Everyday Laptop", "img": "https://placehold.co/150x150/EEE/31343C?text=Laptop"},
        {"name": "Gaming Laptop", "img": "https://placehold.co/150x150/A9EEEF/31343C?text=Gaming+PC"}
    ],
    "Computer Mice": [
        {"name": "Wireless Mouse", "img": "https://placehold.co/150x150/F8F1AE/31343C?text=Mouse"},
        {"name": "Ergonomic Mouse", "img": "https://placehold.co/150x150/AEF8BB/31343C?text=ErgoMouse"}
    ],
    "Keyboards": [
        {"name": "Mechanical Keyboard", "img": "https://placehold.co/150x150/D0AEF8/31343C?text=Mech+Keys"},
        {"name": "Compact Keyboard", "img": "https://placehold.co/150x150/F8AEAE/31343C?text=CompactKeys"}
    ],
    "Monitors": [
        {"name": "Ultrawide Monitor", "img": "https://placehold.co/150x150/AED2F8/31343C?text=Ultrawide"},
        {"name": "Gaming Monitor", "img": "https://placehold.co/150x150/F8CFAE/31343C?text=GamerMon"}
    ],
    "Smartphones": [
        {"name": "Latest Smartphone", "img": "https://placehold.co/150x150/AEF8F1/31343C?text=Phone"},
        {"name": "Budget Phone", "img": "https://placehold.co/150x150/BBAEF8/31343C?text=BudgetPhone"}
    ],
    "Phone Cases": [
        {"name": "Clear Case", "img": "https://placehold.co/150x150/EFAEEF/31343C?text=Case"},
        {"name": "Rugged Case", "img": "https://placehold.co/150x150/F8BBAE/31343C?text=RuggedCase"}
    ],
    "Headphones": [
        {"name": "Noise Cancelling HPs", "img": "https://placehold.co/150x150/AEF8D3/31343C?text=Headphones"},
        {"name": "Wireless Earbuds", "img": "https://placehold.co/150x150/F8AEDB/31343C?text=Earbuds"}
    ],
     "Coffee Makers": [
        {"name": "Drip Coffee Machine", "img": "https://placehold.co/150x150/F8DEA0/31343C?text=DripCoffee"},
        {"name": "Espresso Maker", "img": "https://placehold.co/150x150/A0F8DE/31343C?text=Espresso"}
    ],
    "T-shirts": [
        {"name": "Graphic Tee", "img": "https://placehold.co/150x150/DECAF0/31343C?text=T-Shirt"},
        {"name": "Plain Crew Neck", "img": "https://placehold.co/150x150/A0DEBF/31343C?text=Plain+Tee"}
    ],
    # --- ADD MORE CATEGORIES AND SAMPLES AS NEEDED ---
    # Add entries for categories you frequently see recommended
    # You can find category names in the `product_list.joblib` file
}

def get_sample_products(category_name, count=2):
    """Returns a list of sample products for a given category."""
    samples = SAMPLE_PRODUCT_DATA.get(category_name, [])
    if not samples:
        # Provide a default if category not found
        return [{"name": f"Generic {category_name} Product", "img": "https://placehold.co/150x150/cccccc/31343C?text=Product"}] * count
    # Return up to 'count' samples, shuffling to show variety if more exist
    random.shuffle(samples)
    return samples[:count]