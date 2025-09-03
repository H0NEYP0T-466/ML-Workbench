import pandas as pd
import numpy as np
import random

# Generate House Price dataset
random.seed(42)
np.random.seed(42)

data = {
    "size_sqft": [],
    "bedrooms": [],
    "bathrooms": [],
    "age_years": [],
    "location_score": [],
    "price": []
}

# Generate 550 house records
for i in range(550):
    # Generate realistic house data
    size = random.randint(800, 4000)  # 800-4000 sqft
    bedrooms = random.randint(1, 5)  # 1-5 bedrooms
    bathrooms = random.uniform(1, 3.5)  # 1-3.5 bathrooms
    age = random.randint(0, 50)  # 0-50 years old
    location_score = random.uniform(1, 10)  # 1-10 location rating
    
    # Calculate price based on realistic factors
    base_price = 100000  # Base price
    
    # Size factor (main driver)
    price = base_price + (size * 120)  # $120 per sqft
    
    # Bedroom/bathroom adjustments
    price += bedrooms * 15000
    price += bathrooms * 10000
    
    # Age depreciation
    price -= age * 2000
    
    # Location premium
    price += location_score * 8000
    
    # Add some random variation
    price += random.uniform(-30000, 30000)
    
    # Ensure minimum price
    price = max(price, 80000)
    
    data["size_sqft"].append(size)
    data["bedrooms"].append(bedrooms)
    data["bathrooms"].append(round(bathrooms, 1))
    data["age_years"].append(age)
    data["location_score"].append(round(location_score, 1))
    data["price"].append(round(price, 2))

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv("house_price_dataset.csv", index=False)

print(f"Generated dataset with {len(df)} rows")
print("Sample data:")
print(df.head(10))
print(f"\nPrice statistics:")
print(df["price"].describe())