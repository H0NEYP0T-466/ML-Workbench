import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import seaborn as sns

# DATA
np.random.seed(42)

# Generate synthetic grocery market basket dataset
grocery_items = [
    # Fresh produce
    'apples', 'bananas', 'oranges', 'grapes', 'strawberries', 'tomatoes', 'onions', 'carrots', 'lettuce', 'spinach',
    # Dairy
    'milk', 'cheese', 'yogurt', 'butter', 'cream', 'eggs',
    # Meat & Seafood
    'chicken', 'beef', 'pork', 'fish', 'turkey', 'bacon',
    # Bakery
    'bread', 'bagels', 'muffins', 'croissants', 'pizza_dough',
    # Pantry
    'rice', 'pasta', 'beans', 'flour', 'sugar', 'salt', 'pepper', 'oil', 'vinegar',
    # Beverages
    'coffee', 'tea', 'juice', 'soda', 'water', 'wine', 'beer',
    # Snacks
    'chips', 'cookies', 'crackers', 'nuts', 'candy',
    # Frozen
    'ice_cream', 'frozen_vegetables', 'frozen_pizza'
]

transactions = []
customer_ids = []

print("=== Apriori Market Basket Analysis ===")

# Generate 650 realistic grocery shopping transactions
for i in range(650):
    transaction = []
    customer_id = f'CUST_{i+1:03d}'
    
    # Different shopping patterns based on customer types
    customer_type = np.random.choice(['family', 'single', 'couple', 'student'], 
                                   p=[0.4, 0.25, 0.25, 0.1])
    
    if customer_type == 'family':
        # Family shopping: larger quantities, diverse items
        if np.random.random() < 0.9:  # Staples
            transaction.extend(['milk', 'bread', 'eggs'])
        if np.random.random() < 0.8:  # Fresh produce
            transaction.extend(['apples', 'bananas', 'carrots', 'onions'])
        if np.random.random() < 0.7:  # Meat
            transaction.extend(['chicken', 'beef'])
        if np.random.random() < 0.6:  # Pantry
            transaction.extend(['rice', 'pasta', 'oil'])
        if np.random.random() < 0.5:  # Snacks for kids
            transaction.extend(['cookies', 'juice', 'ice_cream'])
    
    elif customer_type == 'single':
        # Single person: convenience items, smaller quantities
        if np.random.random() < 0.7:  # Basic necessities
            transaction.extend(['milk', 'bread'])
        if np.random.random() < 0.6:  # Quick meals
            transaction.extend(['frozen_pizza', 'pasta'])
        if np.random.random() < 0.5:  # Beverages
            transaction.extend(['coffee', 'beer'])
        if np.random.random() < 0.4:  # Snacks
            transaction.extend(['chips', 'soda'])
    
    elif customer_type == 'couple':
        # Couple: balanced shopping
        if np.random.random() < 0.8:  # Daily essentials
            transaction.extend(['milk', 'eggs', 'bread'])
        if np.random.random() < 0.7:  # Fresh items
            transaction.extend(['tomatoes', 'lettuce', 'cheese'])
        if np.random.random() < 0.6:  # Meat
            transaction.extend(['chicken', 'fish'])
        if np.random.random() < 0.4:  # Wine for dinner
            transaction.extend(['wine', 'pasta'])
    
    else:  # student
        # Student: budget-friendly, simple meals
        if np.random.random() < 0.8:  # Cheap staples
            transaction.extend(['pasta', 'rice', 'beans'])
        if np.random.random() < 0.6:  # Quick energy
            transaction.extend(['coffee', 'bananas'])
        if np.random.random() < 0.5:  # Affordable protein
            transaction.extend(['eggs', 'pork'])
        if np.random.random() < 0.3:  # Occasional treats
            transaction.extend(['cookies', 'soda'])
    
    # Add some seasonal/promotional items
    if np.random.random() < 0.2:
        seasonal_items = ['strawberries', 'grapes', 'turkey', 'muffins']
        transaction.extend(np.random.choice(seasonal_items, size=1))
    
    # Add some random noise
    if np.random.random() < 0.3:
        noise_items = np.random.choice(grocery_items, size=np.random.randint(1, 3), replace=False)
        transaction.extend(noise_items)
    
    # Ensure some correlation patterns
    if 'milk' in transaction and np.random.random() < 0.6:
        transaction.append('cookies')
    if 'bread' in transaction and np.random.random() < 0.5:
        transaction.append('butter')
    if 'pasta' in transaction and np.random.random() < 0.7:
        transaction.append('tomatoes')
    if 'coffee' in transaction and np.random.random() < 0.4:
        transaction.append('sugar')
    
    # Remove duplicates and ensure minimum transaction size
    transaction = list(set(transaction))
    if len(transaction) < 2:
        additional_items = np.random.choice(grocery_items, size=2, replace=False)
        transaction.extend(additional_items)
        transaction = list(set(transaction))
    
    transactions.append(transaction)
    customer_ids.append(customer_id)

print(f"Generated {len(transactions)} grocery transactions")
print(f"Average basket size: {np.mean([len(t) for t in transactions]):.2f}")
print(f"Unique products: {len(set(item for transaction in transactions for item in transaction))}")

# Convert to binary matrix format for Apriori
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
copyData = df.copy()

# Save dataset to CSV
transactions_df = pd.DataFrame({'Customer_ID': customer_ids, 'Items': [','.join(t) for t in transactions]})
transactions_df.to_csv('market_basket_dataset.csv', index=False)

print(f"\nDataset shape: {copyData.shape}")
print(f"Sample grocery baskets:")
for i in range(3):
    items_in_basket = [item for item in te.columns_ if df.iloc[i][item]]
    print(f"Basket {i+1}: {items_in_basket}")

# MODEL
print(f"\n=== Apriori Market Basket Analysis ===")

# Optimized parameters for grocery data
min_support = 0.05  # 5% minimum support (higher for efficiency)
min_confidence = 0.6  # 60% minimum confidence
min_lift = 1.2  # Lift > 1.2 indicates strong positive association

# Generate frequent itemsets using Apriori
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, verbose=1, max_len=4)
print(f"Found {len(frequent_itemsets)} frequent itemsets with min_support={min_support}")

# RULES
# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))
rules_filtered = rules[rules['lift'] >= min_lift].sort_values('lift', ascending=False).reset_index(drop=True)

print(f"Generated {len(rules_filtered)} association rules with min_confidence={min_confidence} and min_lift={min_lift}")

# Print top association rules
print(f"\n=== Top 15 Product Association Rules ===")
print(f"{'#':>3} {'If Customer Buys':>25} {'Then Also Buys':>20} {'Support':>8} {'Confidence':>10} {'Lift':>8}")
print("-" * 90)
for i, rule in rules_filtered.head(15).iterrows():
    antecedent = ', '.join(list(rule['antecedents']))[:23]
    consequent = ', '.join(list(rule['consequents']))[:18]
    print(f"{i+1:>3} {antecedent:>25} {consequent:>20} {rule['support']:>8.4f} {rule['confidence']:>10.4f} {rule['lift']:>8.4f}")

# PLOTS
plt.figure(figsize=(15, 10))

# Plot 1: Bar chart of top 10 frequent items
plt.subplot(2, 3, 1)
item_support = []
for item in te.columns_:
    support = df[item].mean()
    item_support.append((item, support))

item_support.sort(key=lambda x: x[1], reverse=True)
top_10_items = item_support[:10]
items, supports = zip(*top_10_items)

plt.barh(range(len(items)), supports, color='skyblue', alpha=0.8)
plt.yticks(range(len(items)), items)
plt.xlabel('Support')
plt.title('Top 10 Most Frequent Items')
plt.gca().invert_yaxis()

# Plot 2: Network graph of top 10 rules
plt.subplot(2, 3, 2)
if len(rules_filtered) >= 10:
    G = nx.DiGraph()
    top_rules = rules_filtered.head(10)
    
    for _, rule in top_rules.iterrows():
        antecedent = list(rule['antecedents'])[0] if len(rule['antecedents']) == 1 else f"{'&'.join(list(rule['antecedents'])[:2])}"
        consequent = list(rule['consequents'])[0] if len(rule['consequents']) == 1 else f"{'&'.join(list(rule['consequents'])[:2])}"
        
        # Shorten names for better visualization
        ant_short = antecedent.replace('_', ' ')[:8]
        cons_short = consequent.replace('_', ' ')[:8]
        G.add_edge(ant_short, cons_short, weight=rule['lift'])
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    nx.draw(G, pos, with_labels=True, node_color='lightcoral', 
            node_size=800, font_size=8, font_weight='bold',
            edge_color='gray', arrows=True, arrowsize=15)
    plt.title('Top 10 Association Rules Network')

# Plot 3: Support vs Confidence scatter plot
plt.subplot(2, 3, 3)
if len(rules_filtered) > 0:
    scatter = plt.scatter(rules_filtered['support'], rules_filtered['confidence'], 
                         c=rules_filtered['lift'], cmap='viridis', alpha=0.7, s=50)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Rules: Support vs Confidence')
    plt.colorbar(scatter, label='Lift')

# Plot 4: Lift distribution
plt.subplot(2, 3, 4)
if len(rules_filtered) > 0:
    plt.hist(rules_filtered['lift'], bins=15, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(x=1, color='red', linestyle='--', label='Lift = 1 (Independence)')
    plt.xlabel('Lift')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lift Values')
    plt.legend()

# Plot 5: Top rules by confidence
plt.subplot(2, 3, 5)
if len(rules_filtered) >= 8:
    top_conf_rules = rules_filtered.head(8)
    rule_labels = []
    for _, rule in top_conf_rules.iterrows():
        ant = ', '.join(list(rule['antecedents']))[:10]
        cons = ', '.join(list(rule['consequents']))[:10]
        rule_labels.append(f"{ant}→{cons}")
    
    plt.barh(range(len(rule_labels)), top_conf_rules['confidence'], 
             color='lightgreen', alpha=0.8)
    plt.yticks(range(len(rule_labels)), rule_labels, fontsize=8)
    plt.xlabel('Confidence')
    plt.title('Top 8 Rules by Confidence')
    plt.gca().invert_yaxis()

# Plot 6: Transaction size distribution
plt.subplot(2, 3, 6)
transaction_sizes = [len(t) for t in transactions]
plt.hist(transaction_sizes, bins=range(1, max(transaction_sizes)+2), 
         alpha=0.7, color='purple', edgecolor='black')
plt.xlabel('Transaction Size (Number of Items)')
plt.ylabel('Frequency')
plt.title('Transaction Size Distribution')

plt.tight_layout()
plt.savefig('market_basket_apriori_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# CUSTOM INPUT
print("\n=== Your Shopping Cart Analysis ===")
try:
    user_input = input("Enter items in your cart (comma-separated): ").strip()
    if user_input and user_input.lower() not in ['skip', 'default', '']:
        user_items = [item.strip().lower() for item in user_input.split(',')]
        user_items = [item for item in user_items if item in te.columns_]
        
        if user_items:
            print(f"\nYour shopping cart: {user_items}")
            
            # Find product recommendations based on association rules
            recommendations = []
            for _, rule in rules_filtered.iterrows():
                antecedent_items = set(rule['antecedents'])
                consequent_items = set(rule['consequents'])
                user_items_set = set(user_items)
                
                # If user has all antecedent items, recommend consequent items
                if antecedent_items.issubset(user_items_set):
                    for item in consequent_items:
                        if item not in user_items_set:
                            recommendations.append({
                                'item': item,
                                'confidence': rule['confidence'],
                                'lift': rule['lift'],
                                'support': rule['support']
                            })
            
            if recommendations:
                # Sort by confidence and remove duplicates
                seen_items = set()
                print(f"\n=== Recommended Items for You ===")
                for i, rec in enumerate(sorted(recommendations, key=lambda x: x['confidence'], reverse=True)[:8]):
                    if rec['item'] not in seen_items:
                        print(f"{len(seen_items)+1}. {rec['item'].replace('_', ' ').title()}")
                        print(f"   Confidence: {rec['confidence']:.1%}, Lift: {rec['lift']:.2f}")
                        seen_items.add(rec['item'])
            else:
                print("No specific recommendations found. Try adding more common items!")
                # Show some popular combinations
                print("\nPopular combinations include:")
                for _, rule in rules_filtered.head(3).iterrows():
                    items = list(rule['antecedents']) + list(rule['consequents'])
                    print(f"- {', '.join([item.replace('_', ' ').title() for item in items])}")
        else:
            print("No valid grocery items found in your input.")
    else:
        user_items = ['milk', 'bread', 'eggs']  # Default for visualization
        print(f"Using default cart: {user_items}")
        
except (KeyboardInterrupt, EOFError):
    user_items = ['milk', 'bread']  # Default when skipped
    print(f"\nUsing default cart: {user_items}")

print("\nAnalysis complete! Plots saved as 'market_basket_apriori_analysis.png'")