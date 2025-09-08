import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import seaborn as sns

# DATA LOAD
np.random.seed(42)

# Generate synthetic market basket dataset
items = ['bread', 'milk', 'eggs', 'butter', 'cheese', 'yogurt', 'chicken', 'beef', 
         'apples', 'bananas', 'tomatoes', 'onions', 'rice', 'pasta', 'cereal', 
         'coffee', 'tea', 'juice', 'beer', 'wine', 'fish', 'vegetables', 'fruits',
         'meat', 'dairy', 'snacks', 'beverages', 'condiments', 'spices', 'oils']

transactions = []
transaction_ids = []

print("=== FP-Growth Algorithm for Association Rule Learning ===")

# Generate 700 transactions with realistic patterns
for i in range(700):
    transaction = []
    transaction_id = f'T{i+1:03d}'
    
    # Create realistic shopping patterns with higher variety
    if np.random.random() < 0.35:  # Breakfast combo
        if np.random.random() < 0.8:
            transaction.extend(['bread', 'milk'])
        if np.random.random() < 0.6:
            transaction.append('eggs')
        if np.random.random() < 0.4:
            transaction.append('butter')
        if np.random.random() < 0.3:
            transaction.append('coffee')
        if np.random.random() < 0.25:
            transaction.append('cereal')
    
    if np.random.random() < 0.3:  # Dinner combo
        if np.random.random() < 0.7:
            transaction.extend(['chicken', 'rice'])
        if np.random.random() < 0.5:
            transaction.extend(['onions', 'tomatoes'])
        if np.random.random() < 0.3:
            transaction.append('beef')
        if np.random.random() < 0.4:
            transaction.append('vegetables')
    
    if np.random.random() < 0.25:  # Healthy combo
        if np.random.random() < 0.8:
            transaction.extend(['apples', 'bananas'])
        if np.random.random() < 0.6:
            transaction.extend(['yogurt', 'milk'])
        if np.random.random() < 0.4:
            transaction.append('fruits')
    
    if np.random.random() < 0.2:  # Party combo
        if np.random.random() < 0.9:
            transaction.extend(['beer', 'wine'])
        if np.random.random() < 0.7:
            transaction.append('cheese')
        if np.random.random() < 0.5:
            transaction.append('snacks')
        if np.random.random() < 0.3:
            transaction.append('beverages')
    
    if np.random.random() < 0.15:  # Cooking combo
        if np.random.random() < 0.8:
            transaction.extend(['onions', 'tomatoes', 'oils'])
        if np.random.random() < 0.6:
            transaction.append('spices')
        if np.random.random() < 0.4:
            transaction.append('condiments')
    
    # Add random items to increase variety
    num_random = np.random.poisson(2.5)
    random_items = np.random.choice(items, size=min(num_random, 5), replace=False)
    transaction.extend(random_items)
    
    # Remove duplicates and filter empty
    transaction = list(set(transaction))
    if len(transaction) > 0:
        transactions.append(transaction)
        transaction_ids.append(transaction_id)

print(f"Generated {len(transactions)} transactions")
print(f"Average transaction size: {np.mean([len(t) for t in transactions]):.2f}")
print(f"Unique items: {len(set(item for transaction in transactions for item in transaction))}")

# Convert to binary matrix format for FP-Growth
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
copyData = df.copy()

print(f"\nDataset shape: {copyData.shape}")
print(f"Sample transactions:")
for i in range(3):
    items_in_transaction = [item for item in te.columns_ if df.iloc[i][item]]
    print(f"Transaction {i+1}: {items_in_transaction}")

# MODEL
print(f"\n=== FP-Growth Algorithm ===")

# FP-Growth parameters
min_support = 0.05  # 5% minimum support
min_confidence = 0.6  # 60% minimum confidence
min_lift = 1.1  # Lift > 1.1 indicates positive association

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
print(f"\n=== FP-Growth Results ===")
print(f"Total frequent itemsets: {len(frequent_itemsets)}")
print(f"Total association rules: {len(rules_filtered)}")
print(f"Average support: {rules_filtered['support'].mean():.4f}")
print(f"Average confidence: {rules_filtered['confidence'].mean():.4f}")
print(f"Average lift: {rules_filtered['lift'].mean():.4f}")

print(f"\n=== Top 10 Association Rules ===")
print(f"{'#':>3} {'Antecedent':>20} {'->':>3} {'Consequent':>15} {'Support':>8} {'Confidence':>10} {'Lift':>8}")
print("-" * 80)
for i, rule in rules_filtered.head(10).iterrows():
    antecedent = ', '.join(list(rule['antecedents']))[:18]
    consequent = ', '.join(list(rule['consequents']))[:13]
    print(f"{i+1:>3} {antecedent:>20} {'->':>3} {consequent:>15} {rule['support']:>8.4f} {rule['confidence']:>10.4f} {rule['lift']:>8.4f}")

# CUSTOM INPUT
print("\n=== Custom Shopping Cart Analysis ===")
try:
    user_input = input("Enter purchased items (comma-separated): ").strip()
    if user_input and user_input.lower() not in ['skip', 'default', '']:
        user_items = [item.strip().lower() for item in user_input.split(',')]
        user_items = [item for item in user_items if item in te.columns_]
        
        if user_items:
            print(f"\nYour cart: {user_items}")
            
            # Find rules that apply to user's items
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
            
            # Sort recommendations by confidence * lift
            recommendations = sorted(recommendations, 
                                   key=lambda x: x['confidence'] * x['lift'], 
                                   reverse=True)
            
            print(f"\n=== Recommended Items ===")
            if recommendations:
                seen_items = set()
                for i, rec in enumerate(recommendations[:5]):
                    if rec['item'] not in seen_items:
                        print(f"{i+1}. {rec['item']} (confidence: {rec['confidence']:.3f}, lift: {rec['lift']:.3f})")
                        seen_items.add(rec['item'])
            else:
                print("No specific recommendations found based on association rules.")
        else:
            print("No valid items found in your input.")
    else:
        user_items = ['bread', 'milk']  # Default for visualization
        print(f"Using default cart: {user_items}")
        
except (KeyboardInterrupt, EOFError):
    user_items = ['bread', 'milk']  # Default for visualization
    print(f"\nUsing default cart: {user_items}")

# PLOTS
plt.figure(figsize=(15, 12))

# Plot 1: Item frequency
plt.subplot(3, 3, 1)
item_support = {}
for item in te.columns_:
    item_support[item] = df[item].mean()

sorted_items = sorted(item_support.items(), key=lambda x: x[1], reverse=True)
item_names, supports = zip(*sorted_items[:15])

plt.barh(range(len(item_names)), supports, color='lightblue', alpha=0.7)
plt.yticks(range(len(item_names)), item_names)
plt.xlabel('Support')
plt.title('Top 15 Items by Support')
plt.gca().invert_yaxis()

# Plot 2: Support vs Confidence scatter
plt.subplot(3, 3, 2)
if len(rules_filtered) > 0:
    scatter = plt.scatter(rules_filtered['support'], rules_filtered['confidence'], 
                         c=rules_filtered['lift'], cmap='viridis', alpha=0.7, s=50)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence (colored by Lift)')
    plt.colorbar(scatter, label='Lift')

# Plot 3: Lift distribution
plt.subplot(3, 3, 3)
if len(rules_filtered) > 0:
    plt.hist(rules_filtered['lift'], bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(x=1, color='red', linestyle='--', label='Lift = 1 (Independence)')
    plt.xlabel('Lift')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lift Values')
    plt.legend()

# Plot 4: Frequent itemsets by size
plt.subplot(3, 3, 4)
itemset_sizes = [len(itemset) for itemset in frequent_itemsets['itemsets']]
size_counts = pd.Series(itemset_sizes).value_counts().sort_index()
plt.bar(size_counts.index, size_counts.values, color='orange', alpha=0.7)
plt.xlabel('Itemset Size')
plt.ylabel('Count')
plt.title('Frequent Itemsets by Size')

# Plot 5: Network graph of top associations
plt.subplot(3, 3, 5)
if len(rules_filtered) >= 5:
    G = nx.DiGraph()
    top_rules_graph = rules_filtered.head(8)  # Top 8 for better visualization
    
    for _, rule in top_rules_graph.iterrows():
        antecedent = ', '.join(list(rule['antecedents']))
        consequent = ', '.join(list(rule['consequents']))
        G.add_edge(antecedent, consequent, weight=rule['lift'])
    
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=8, font_weight='bold',
            edge_color='gray', arrows=True, arrowsize=20)
    plt.title('Association Rules Network')

# Plot 6: Confidence vs Lift scatter
plt.subplot(3, 3, 6)
if len(rules_filtered) > 0:
    scatter = plt.scatter(rules_filtered['confidence'], rules_filtered['lift'], 
                         c=rules_filtered['support'], cmap='plasma', alpha=0.7, s=50)
    plt.xlabel('Confidence')
    plt.ylabel('Lift')
    plt.title('Confidence vs Lift (colored by Support)')
    plt.colorbar(scatter, label='Support')

# Plot 7: Top rules by lift
plt.subplot(3, 3, 7)
if len(rules_filtered) >= 10:
    top_rules = rules_filtered.head(10)
    rule_labels = [f"{', '.join(list(rule['antecedents']))} → {', '.join(list(rule['consequents']))}"[:25] 
                   for _, rule in top_rules.iterrows()]
    
    plt.barh(range(len(rule_labels)), top_rules['lift'], color='purple', alpha=0.7)
    plt.yticks(range(len(rule_labels)), rule_labels)
    plt.xlabel('Lift')
    plt.title('Top 10 Rules by Lift')
    plt.gca().invert_yaxis()

# Plot 8: Algorithm efficiency comparison
plt.subplot(3, 3, 8)
algorithms = ['Apriori', 'FP-Growth']
efficiency = [0.7, 0.95]  # FP-Growth is generally more efficient
memory_usage = [0.8, 0.6]  # FP-Growth uses less memory

x = np.arange(len(algorithms))
width = 0.35

plt.bar(x - width/2, efficiency, width, label='Speed', alpha=0.7, color='blue')
plt.bar(x + width/2, memory_usage, width, label='Memory Efficiency', alpha=0.7, color='orange')
plt.ylabel('Efficiency Score')
plt.title('Algorithm Comparison')
plt.xticks(x, algorithms)
plt.legend()
plt.ylim(0, 1)

# Plot 9: Transaction size distribution
plt.subplot(3, 3, 9)
transaction_sizes = [len(transaction) for transaction in transactions]
plt.hist(transaction_sizes, bins=range(1, max(transaction_sizes)+2), 
         alpha=0.7, color='brown', edgecolor='black')
plt.xlabel('Transaction Size')
plt.ylabel('Frequency')
plt.title('Transaction Size Distribution')
plt.xticks(range(1, max(transaction_sizes)+1, 2))

plt.tight_layout()
plt.savefig('fpgrowth_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Plot saved as 'fpgrowth_analysis.png'")