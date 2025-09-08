import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import seaborn as sns

# DATA LOAD
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

print("=== FP-Growth for Market Basket Analysis ===")

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
        if np.random.random() < 0.5:  # Beverages
            transaction.extend(['juice', 'coffee'])
        if np.random.random() < 0.4:  # Snacks
            transaction.extend(['chips', 'cookies'])
        if np.random.random() < 0.3:  # Frozen
            transaction.append('ice_cream')
    
    elif customer_type == 'single':
        # Single person: convenience items, smaller portions
        if np.random.random() < 0.7:  # Quick meals
            transaction.extend(['frozen_pizza', 'pasta'])
        if np.random.random() < 0.6:  # Basic staples
            transaction.extend(['milk', 'bread'])
        if np.random.random() < 0.5:  # Coffee/tea
            transaction.append('coffee')
        if np.random.random() < 0.4:  # Snacks
            transaction.extend(['chips', 'crackers'])
        if np.random.random() < 0.3:  # Beverages
            transaction.extend(['soda', 'beer'])
        if np.random.random() < 0.3:  # Easy proteins
            transaction.append('chicken')
    
    elif customer_type == 'couple':
        # Couple: quality items, cooking ingredients
        if np.random.random() < 0.8:  # Cooking basics
            transaction.extend(['onions', 'tomatoes', 'oil'])
        if np.random.random() < 0.7:  # Fresh ingredients
            transaction.extend(['chicken', 'fish'])
        if np.random.random() < 0.6:  # Quality dairy
            transaction.extend(['cheese', 'butter'])
        if np.random.random() < 0.5:  # Wine for dinner
            transaction.append('wine')
        if np.random.random() < 0.4:  # Fresh produce
            transaction.extend(['lettuce', 'tomatoes'])
        if np.random.random() < 0.3:  # Pantry items
            transaction.extend(['rice', 'pasta'])
    
    else:  # student
        # Student: budget items, easy prep
        if np.random.random() < 0.8:  # Cheap carbs
            transaction.extend(['pasta', 'bread'])
        if np.random.random() < 0.6:  # Protein
            transaction.extend(['eggs', 'peanut_butter'])
        if np.random.random() < 0.5:  # Beverages
            transaction.extend(['coffee', 'soda'])
        if np.random.random() < 0.4:  # Snacks
            transaction.extend(['chips', 'cookies'])
        if np.random.random() < 0.3:  # Basics
            transaction.append('milk')
    
    # Add some random items to create noise
    num_random = np.random.poisson(1.5)
    random_items = np.random.choice(grocery_items, size=min(num_random, 3), replace=False)
    transaction.extend(random_items)
    
    # Remove duplicates and filter empty
    transaction = list(set(transaction))
    if len(transaction) > 0:
        transactions.append(transaction)
        customer_ids.append(customer_id)

print(f"Generated {len(transactions)} grocery transactions")
print(f"Average basket size: {np.mean([len(t) for t in transactions]):.2f}")
print(f"Unique products: {len(set(item for transaction in transactions for item in transaction))}")

# Convert to binary matrix format for FP-Growth
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
copyData = df.copy()

print(f"\nDataset shape: {copyData.shape}")
print(f"Sample grocery baskets:")
for i in range(3):
    items_in_basket = [item for item in te.columns_ if df.iloc[i][item]]
    print(f"Basket {i+1}: {items_in_basket}")

# MODEL
print(f"\n=== FP-Growth Market Basket Analysis ===")

# Optimized parameters for grocery data
min_support = 0.03  # 3% minimum support (lower for grocery data)
min_confidence = 0.5  # 50% minimum confidence
min_lift = 1.2  # Lift > 1.2 indicates strong positive association

# Generate frequent itemsets using FP-Growth
frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True, verbose=1)
print(f"Found {len(frequent_itemsets)} frequent itemsets with min_support={min_support}")

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))

# Filter rules by lift
rules_filtered = rules[rules['lift'] >= min_lift].copy()
rules_filtered = rules_filtered.sort_values('lift', ascending=False).reset_index(drop=True)

print(f"Generated {len(rules_filtered)} association rules with min_confidence={min_confidence} and min_lift={min_lift}")

# METRICS
print(f"\n=== Market Basket Analysis Results ===")
print(f"Total frequent itemsets: {len(frequent_itemsets)}")
print(f"Total association rules: {len(rules_filtered)}")
print(f"Average support: {rules_filtered['support'].mean():.4f}")
print(f"Average confidence: {rules_filtered['confidence'].mean():.4f}")
print(f"Average lift: {rules_filtered['lift'].mean():.4f}")

print(f"\n=== Top 15 Product Association Rules ===")
print(f"{'#':>3} {'If Customer Buys':>25} {'Then Also Buys':>20} {'Support':>8} {'Confidence':>10} {'Lift':>8}")
print("-" * 90)
for i, rule in rules_filtered.head(15).iterrows():
    antecedent = ', '.join(list(rule['antecedents']))[:23]
    consequent = ', '.join(list(rule['consequents']))[:18]
    print(f"{i+1:>3} {antecedent:>25} {consequent:>20} {rule['support']:>8.4f} {rule['confidence']:>10.4f} {rule['lift']:>8.4f}")

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
                                'support': rule['support'],
                                'based_on': list(antecedent_items)
                            })
            
            # Sort recommendations by confidence * lift
            recommendations = sorted(recommendations, 
                                   key=lambda x: x['confidence'] * x['lift'], 
                                   reverse=True)
            
            print(f"\n=== Recommended Products ===")
            if recommendations:
                seen_items = set()
                for i, rec in enumerate(recommendations[:8]):
                    if rec['item'] not in seen_items:
                        print(f"{i+1}. {rec['item'].title().replace('_', ' ')}")
                        print(f"   Based on: {[item.replace('_', ' ') for item in rec['based_on']]}")
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
    user_items = ['milk', 'bread', 'eggs']  # Default for visualization
    print(f"\nUsing default cart: {user_items}")

# PLOTS
plt.figure(figsize=(15, 12))

# Plot 1: Top selling products
plt.subplot(3, 3, 1)
item_support = {}
for item in te.columns_:
    item_support[item] = df[item].mean()

sorted_items = sorted(item_support.items(), key=lambda x: x[1], reverse=True)
item_names, supports = zip(*sorted_items[:15])
display_names = [name.replace('_', ' ').title() for name in item_names]

plt.barh(range(len(display_names)), supports, color='lightgreen', alpha=0.7)
plt.yticks(range(len(display_names)), display_names)
plt.xlabel('Purchase Frequency')
plt.title('Top 15 Products by Popularity')
plt.gca().invert_yaxis()

# Plot 2: Market basket rules visualization
plt.subplot(3, 3, 2)
if len(rules_filtered) > 0:
    scatter = plt.scatter(rules_filtered['support'], rules_filtered['confidence'], 
                         c=rules_filtered['lift'], cmap='viridis', alpha=0.7, s=60)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Market Basket Rules')
    plt.colorbar(scatter, label='Lift')

# Plot 3: Product categories analysis
plt.subplot(3, 3, 3)
categories = {
    'Produce': ['apples', 'bananas', 'oranges', 'tomatoes', 'onions', 'carrots', 'lettuce'],
    'Dairy': ['milk', 'cheese', 'yogurt', 'butter', 'eggs'],
    'Meat': ['chicken', 'beef', 'pork', 'fish', 'bacon'],
    'Beverages': ['coffee', 'tea', 'juice', 'soda', 'wine', 'beer'],
    'Pantry': ['rice', 'pasta', 'bread', 'oil', 'flour', 'sugar']
}

category_sales = {}
for category, items in categories.items():
    category_sales[category] = sum(df[item].sum() for item in items if item in df.columns)

plt.pie(category_sales.values(), labels=category_sales.keys(), autopct='%1.1f%%', 
        colors=['lightcoral', 'lightskyblue', 'lightgreen', 'gold', 'plum'])
plt.title('Sales by Product Category')

# Plot 4: Basket size distribution
plt.subplot(3, 3, 4)
basket_sizes = [len(transaction) for transaction in transactions]
plt.hist(basket_sizes, bins=range(1, max(basket_sizes)+2), 
         alpha=0.7, color='orange', edgecolor='black')
plt.xlabel('Basket Size (Number of Items)')
plt.ylabel('Number of Customers')
plt.title('Shopping Basket Size Distribution')

# Plot 5: Association strength network
plt.subplot(3, 3, 5)
if len(rules_filtered) >= 5:
    G = nx.DiGraph()
    top_rules_graph = rules_filtered.head(10)  # Top 10 for better visualization
    
    for _, rule in top_rules_graph.iterrows():
        for ant in rule['antecedents']:
            for cons in rule['consequents']:
                G.add_edge(ant.replace('_', ' '), cons.replace('_', ' '), 
                          weight=rule['lift'])
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=800, font_size=8, font_weight='bold',
            edge_color='gray', arrows=True, arrowsize=20)
    plt.title('Product Association Network')

# Plot 6: Top rules by confidence
plt.subplot(3, 3, 6)
if len(rules_filtered) >= 10:
    top_confidence = rules_filtered.nlargest(10, 'confidence')
    rule_labels = []
    for _, rule in top_confidence.iterrows():
        ant = ', '.join([item.replace('_', ' ') for item in rule['antecedents']])[:15]
        cons = ', '.join([item.replace('_', ' ') for item in rule['consequents']])[:15]
        rule_labels.append(f"{ant} → {cons}")
    
    plt.barh(range(len(rule_labels)), top_confidence['confidence'], color='purple', alpha=0.7)
    plt.yticks(range(len(rule_labels)), rule_labels)
    plt.xlabel('Confidence')
    plt.title('Top 10 Rules by Confidence')
    plt.gca().invert_yaxis()

# Plot 7: Lift vs Support colored by confidence
plt.subplot(3, 3, 7)
if len(rules_filtered) > 0:
    scatter = plt.scatter(rules_filtered['lift'], rules_filtered['support'], 
                         c=rules_filtered['confidence'], cmap='plasma', alpha=0.7, s=60)
    plt.xlabel('Lift')
    plt.ylabel('Support')
    plt.title('Rule Strength Analysis')
    plt.colorbar(scatter, label='Confidence')

# Plot 8: Customer type shopping patterns (simulated)
plt.subplot(3, 3, 8)
customer_types = ['Family', 'Single', 'Couple', 'Student']
avg_basket_size = [12, 6, 8, 5]  # Simulated based on our generation logic
avg_spending = [150, 45, 85, 35]  # Simulated

x = np.arange(len(customer_types))
width = 0.35

fig_ax = plt.gca()
ax2 = fig_ax.twinx()

bars1 = fig_ax.bar(x - width/2, avg_basket_size, width, label='Avg Basket Size', 
                   alpha=0.7, color='blue')
bars2 = ax2.bar(x + width/2, avg_spending, width, label='Avg Spending ($)', 
                alpha=0.7, color='red')

fig_ax.set_xlabel('Customer Type')
fig_ax.set_ylabel('Average Basket Size', color='blue')
ax2.set_ylabel('Average Spending ($)', color='red')
fig_ax.set_title('Shopping Patterns by Customer Type')
fig_ax.set_xticks(x)
fig_ax.set_xticklabels(customer_types)

# Plot 9: Item frequency heatmap (top items)
plt.subplot(3, 3, 9)
top_items = sorted_items[:10]
item_matrix = []
for item1, _ in top_items:
    row = []
    for item2, _ in top_items:
        # Co-occurrence frequency
        co_occurrence = (df[item1] & df[item2]).sum() / len(df)
        row.append(co_occurrence)
    item_matrix.append(row)

item_labels = [item.replace('_', ' ').title() for item, _ in top_items]
sns.heatmap(item_matrix, xticklabels=item_labels, yticklabels=item_labels, 
            annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Co-occurrence'})
plt.title('Product Co-occurrence Heatmap')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('market_basket_fpgrowth.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nMarket basket analysis complete! Plot saved as 'market_basket_fpgrowth.png'")