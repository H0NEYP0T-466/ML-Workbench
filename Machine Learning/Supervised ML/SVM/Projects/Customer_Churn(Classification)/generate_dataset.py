import pandas as pd
import numpy as np
import random

# Generate Customer Churn dataset
random.seed(42)
np.random.seed(42)

data = {
    "monthly_charges": [],
    "total_charges": [],
    "tenure_months": [],
    "customer_service_calls": [],
    "data_usage_gb": [],
    "churn": []
}

# Generate 550 customer records
for i in range(550):
    # Generate realistic customer data
    tenure = random.randint(1, 72)  # 1-72 months
    monthly_charges = random.uniform(30, 120)  # $30-120 per month
    total_charges = monthly_charges * tenure + random.uniform(-200, 200)  # Some variation
    service_calls = random.randint(0, 8)  # 0-8 service calls
    data_usage = random.uniform(0.5, 50)  # 0.5-50 GB usage
    
    # Determine churn based on realistic factors
    churn_probability = 0.1  # Base probability
    
    # Higher churn if:
    if tenure < 12:  # New customers
        churn_probability += 0.3
    if monthly_charges > 80:  # High charges
        churn_probability += 0.2
    if service_calls > 3:  # Many service issues
        churn_probability += 0.4
    if data_usage < 2:  # Low usage
        churn_probability += 0.2
    
    churn = 1 if random.random() < churn_probability else 0
    
    data["monthly_charges"].append(round(monthly_charges, 2))
    data["total_charges"].append(round(total_charges, 2))
    data["tenure_months"].append(tenure)
    data["customer_service_calls"].append(service_calls)
    data["data_usage_gb"].append(round(data_usage, 2))
    data["churn"].append("Yes" if churn else "No")

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv("customer_churn_dataset.csv", index=False)

print(f"Generated dataset with {len(df)} rows")
print("Sample data:")
print(df.head(10))
print(f"\nChurn distribution:")
print(df["churn"].value_counts())