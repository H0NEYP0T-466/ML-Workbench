import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import networkx as nx
import seaborn as sns

# DATA LOAD
np.random.seed(42)

# Generate synthetic market basket dataset
items = ['bread', 'milk', 'eggs', 'butter', 'cheese', 'yogurt', 'chicken', 'beef', 
         'apples', 'bananas', 'tomatoes', 'onions', 'rice', 'pasta', 'cereal', 
         'coffee', 'tea', 'juice', 'beer', 'wine']

transactions = []
transaction_ids = []

print("=== Apriori Algorithm for Association Rule Learning ===")

# Generate 600 transactions with realistic patterns
for i in range(600):
    transaction = []
    transaction_id = f'T{i+1:03d}'
    
    # Create realistic shopping patterns
    if np.random.random() < 0.3:  # Breakfast combo
        if np.random.random() < 0.8:
            transaction.extend(['bread', 'milk'])
        if np.random.random() < 0.6:
            transaction.append('eggs')
        if np.random.random() < 0.4:
            transaction.append('butter')
        if np.random.random() < 0.3:
            transaction.append('coffee')
    
    if np.random.random() < 0.25:  # Dinner combo
        if np.random.random() < 0.7:
            transaction.extend(['chicken', 'rice'])
        if np.random.random() < 0.5:
            transaction.extend(['onions', 'tomatoes'])
        if np.random.random() < 0.3:
            transaction.append('beef')
    
    if np.random.random() < 0.2:  # Healthy combo
        if np.random.random() < 0.8:
            transaction.extend(['apples', 'bananas'])
        if np.random.random() < 0.6:
            transaction.extend(['yogurt', 'milk'])
    
    if np.random.random() < 0.15:  # Party combo
        if np.random.random() < 0.9:
            transaction.extend(['beer', 'cheese'])
        if np.random.random() < 0.4:
            transaction.append('wine')
    
    # Add random items
    num_random = np.random.poisson(2)
    random_items = np.random.choice(items, size=min(num_random, 3), replace=False)
    transaction.extend(random_items)
    
    # Remove duplicates and ensure minimum transaction size
    transaction = list(set(transaction))
    if len(transaction) < 2:
        additional_items = np.random.choice(items, size=2, replace=False)
        transaction.extend(additional_items)
        transaction = list(set(transaction))
    
    transactions.append(transaction)
    transaction_ids.append(transaction_id)

print(f"Generated {len(transactions)} transactions")
print(f"Unique items: {len(items)}")
print(f"Average transaction size: {np.mean([len(t) for t in transactions]):.2f}")

# Convert to DataFrame format
# Create binary matrix representation
df_binary = pd.DataFrame(False, index=transaction_ids, columns=items)
for i, transaction in enumerate(transactions):
    for item in transaction:
        df_binary.loc[transaction_ids[i], item] = True

copyData = df_binary.copy()

print(f"\nDataset shape: {copyData.shape}")
print(f"First few transactions:")
for i in range(3):
    items_in_transaction = [item for item in items if copyData.iloc[i][item]]
    print(f"{transaction_ids[i]}: {items_in_transaction}")

# APRIORI ALGORITHM IMPLEMENTATION
def calculate_support(itemset, transactions_binary):
    """Calculate support for an itemset"""
    if isinstance(itemset, str):
        itemset = [itemset]
    
    support_count = 0
    for _, transaction in transactions_binary.iterrows():
        if all(transaction[item] for item in itemset):
            support_count += 1
    
    return support_count / len(transactions_binary)

def generate_frequent_itemsets(transactions_binary, min_support=0.05):
    """Generate frequent itemsets using Apriori algorithm"""
    items = list(transactions_binary.columns)
    frequent_itemsets = {}
    
    # Level 1: Individual items
    print(f"\n=== Generating Frequent Itemsets (min_support={min_support}) ===")
    level_1 = {}
    for item in items:
        support = calculate_support([item], transactions_binary)
        if support >= min_support:
            level_1[frozenset([item])] = support
    
    frequent_itemsets[1] = level_1
    print(f"Level 1: {len(level_1)} frequent items")
    
    # Generate higher level itemsets
    k = 2
    while frequent_itemsets[k-1]:
        level_k = {}
        prev_itemsets = list(frequent_itemsets[k-1].keys())
        
        # Generate candidates
        candidates = set()
        for i in range(len(prev_itemsets)):
            for j in range(i+1, len(prev_itemsets)):
                union = prev_itemsets[i] | prev_itemsets[j]
                if len(union) == k:
                    candidates.add(union)
        
        # Check support for candidates
        for candidate in candidates:
            support = calculate_support(list(candidate), transactions_binary)
            if support >= min_support:
                level_k[candidate] = support
        
        if level_k:
            frequent_itemsets[k] = level_k
            print(f"Level {k}: {len(level_k)} frequent itemsets")
            k += 1
        else:
            break
    
    return frequent_itemsets

def generate_association_rules(frequent_itemsets, min_confidence=0.6):
    """Generate association rules from frequent itemsets"""
    rules = []
    
    print(f"\n=== Generating Association Rules (min_confidence={min_confidence}) ===")
    
    for k in range(2, len(frequent_itemsets) + 1):
        for itemset, support in frequent_itemsets[k].items():
            itemset_list = list(itemset)
            
            # Generate all possible antecedent-consequent pairs
            for i in range(1, len(itemset_list)):
                for antecedent in combinations(itemset_list, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    if antecedent in frequent_itemsets[len(antecedent)]:
                        antecedent_support = frequent_itemsets[len(antecedent)][antecedent]
                        confidence = support / antecedent_support
                        
                        if confidence >= min_confidence:
                            lift = confidence / calculate_support(list(consequent), copyData)
                            
                            rules.append({
                                'antecedent': list(antecedent),
                                'consequent': list(consequent),
                                'support': support,
                                'confidence': confidence,
                                'lift': lift,
                                'antecedent_support': antecedent_support,
                                'consequent_support': calculate_support(list(consequent), copyData)
                            })
    
    return sorted(rules, key=lambda x: x['confidence'], reverse=True)

# MODEL
min_support = 0.05
min_confidence = 0.6

frequent_itemsets = generate_frequent_itemsets(copyData, min_support)
association_rules = generate_association_rules(frequent_itemsets, min_confidence)

# METRICS
print(f"\n=== Apriori Results ===")
print(f"Total frequent itemsets: {sum(len(level) for level in frequent_itemsets.values())}")
print(f"Total association rules: {len(association_rules)}")

print(f"\n=== Top 10 Association Rules ===")
for i, rule in enumerate(association_rules[:10]):
    antecedent_str = ', '.join(rule['antecedent'])
    consequent_str = ', '.join(rule['consequent'])
    print(f"{i+1}. {antecedent_str} → {consequent_str}")
    print(f"   Support: {rule['support']:.4f}, Confidence: {rule['confidence']:.4f}, Lift: {rule['lift']:.4f}")

# PLOTS
plt.figure(figsize=(15, 12))

# Plot 1: Item frequency
plt.subplot(3, 3, 1)
item_support = {item: calculate_support([item], copyData) for item in items}
sorted_items = sorted(item_support.items(), key=lambda x: x[1], reverse=True)
item_names, supports = zip(*sorted_items[:15])

plt.barh(range(len(item_names)), supports, color='skyblue', alpha=0.7)
plt.yticks(range(len(item_names)), item_names)
plt.xlabel('Support')
plt.title('Top 15 Items by Support')
plt.gca().invert_yaxis()

# Plot 2: Support vs Confidence scatter
plt.subplot(3, 3, 2)
if association_rules:
    supports = [rule['support'] for rule in association_rules]
    confidences = [rule['confidence'] for rule in association_rules]
    lifts = [rule['lift'] for rule in association_rules]
    
    scatter = plt.scatter(supports, confidences, c=lifts, cmap='viridis', alpha=0.7)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Association Rules: Support vs Confidence')
    plt.colorbar(scatter, label='Lift')

# Plot 3: Top rules by confidence
plt.subplot(3, 3, 3)
if len(association_rules) >= 10:
    top_rules = association_rules[:10]
    rule_labels = [f"{', '.join(rule['antecedent'][:2])}→{', '.join(rule['consequent'][:2])}" for rule in top_rules]
    confidences = [rule['confidence'] for rule in top_rules]
    
    plt.barh(range(len(rule_labels)), confidences, color='orange', alpha=0.7)
    plt.yticks(range(len(rule_labels)), rule_labels, fontsize=8)
    plt.xlabel('Confidence')
    plt.title('Top 10 Rules by Confidence')
    plt.gca().invert_yaxis()

# Plot 4: Lift distribution
plt.subplot(3, 3, 4)
if association_rules:
    lifts = [rule['lift'] for rule in association_rules]
    plt.hist(lifts, bins=20, alpha=0.7, color='green')
    plt.axvline(x=1, color='red', linestyle='--', label='Lift = 1 (Independence)')
    plt.xlabel('Lift')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lift Values')
    plt.legend()

# Plot 5: Network graph of top associations
plt.subplot(3, 3, 5)
if len(association_rules) >= 5:
    G = nx.DiGraph()
    top_rules_graph = association_rules[:8]  # Top 8 for better visualization
    
    for rule in top_rules_graph:
        antecedent_str = ', '.join(rule['antecedent'])
        consequent_str = ', '.join(rule['consequent'])
        G.add_edge(antecedent_str, consequent_str, weight=rule['confidence'])
    
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1000, font_size=8, font_weight='bold',
            edge_color='gray', arrows=True, arrowsize=20)
    plt.title('Association Rules Network')

# Plot 6: Itemset size distribution
plt.subplot(3, 3, 6)
itemset_sizes = []
for level, itemsets in frequent_itemsets.items():
    itemset_sizes.extend([level] * len(itemsets))

if itemset_sizes:
    unique_sizes = sorted(set(itemset_sizes))
    size_counts = [itemset_sizes.count(size) for size in unique_sizes]
    plt.bar(unique_sizes, size_counts, color='purple', alpha=0.7)
    plt.xlabel('Itemset Size')
    plt.ylabel('Number of Frequent Itemsets')
    plt.title('Frequent Itemsets by Size')

# CUSTOM INPUT
print("\n=== Custom Market Basket Prediction ===")
print("Enter items in your basket (comma-separated):")
user_basket_input = input("Your basket: ").strip()
user_basket = [item.strip().lower() for item in user_basket_input.split(',') if item.strip()]

print(f"\nYour basket: {user_basket}")

# Find applicable rules
applicable_rules = []
for rule in association_rules:
    antecedent_lower = [item.lower() for item in rule['antecedent']]
    if all(item in user_basket for item in antecedent_lower):
        applicable_rules.append(rule)

if applicable_rules:
    print(f"\n=== Recommendations based on Association Rules ===")
    print(f"Found {len(applicable_rules)} applicable rules:")
    
    recommended_items = {}
    for i, rule in enumerate(applicable_rules[:5]):  # Top 5 recommendations
        antecedent_str = ', '.join(rule['antecedent'])
        consequent_str = ', '.join(rule['consequent'])
        print(f"{i+1}. Since you have {antecedent_str} → consider {consequent_str}")
        print(f"   Confidence: {rule['confidence']:.2%}, Lift: {rule['lift']:.2f}")
        
        for item in rule['consequent']:
            if item.lower() not in user_basket:
                if item in recommended_items:
                    recommended_items[item] += rule['confidence']
                else:
                    recommended_items[item] = rule['confidence']
    
    if recommended_items:
        sorted_recommendations = sorted(recommended_items.items(), key=lambda x: x[1], reverse=True)
        print(f"\n=== Top Recommended Items ===")
        for item, total_confidence in sorted_recommendations[:5]:
            print(f"• {item} (total confidence: {total_confidence:.2%})")
else:
    print("No specific recommendations found for your basket.")
    print("Consider popular items:", [item for item, _ in sorted_items[:5]])

# Plot 7: User basket visualization
plt.subplot(3, 3, 7)
if user_basket and association_rules:
    # Show recommendations
    applicable_confidences = [rule['confidence'] for rule in applicable_rules[:10]]
    applicable_labels = [f"{', '.join(rule['consequent'])}" for rule in applicable_rules[:10]]
    
    if applicable_confidences:
        plt.barh(range(len(applicable_labels)), applicable_confidences, color='red', alpha=0.7)
        plt.yticks(range(len(applicable_labels)), applicable_labels, fontsize=8)
        plt.xlabel('Confidence')
        plt.title('Recommendations for Your Basket')
        plt.gca().invert_yaxis()
    else:
        plt.text(0.5, 0.5, 'No specific\nrecommendations\nfound', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title('Recommendations for Your Basket')

# Plot 8: Support-Confidence heatmap for top item pairs
plt.subplot(3, 3, 8)
if len(association_rules) >= 5:
    # Create matrix for top items
    top_items = [item for item, _ in sorted_items[:8]]
    matrix_size = len(top_items)
    conf_matrix = np.zeros((matrix_size, matrix_size))
    
    for rule in association_rules:
        if (len(rule['antecedent']) == 1 and len(rule['consequent']) == 1 and
            rule['antecedent'][0] in top_items and rule['consequent'][0] in top_items):
            i = top_items.index(rule['antecedent'][0])
            j = top_items.index(rule['consequent'][0])
            conf_matrix[i, j] = rule['confidence']
    
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=top_items, yticklabels=top_items)
    plt.title('Confidence Matrix (Top Items)')
    plt.ylabel('Antecedent')
    plt.xlabel('Consequent')

# Plot 9: Transaction size distribution
plt.subplot(3, 3, 9)
transaction_sizes = [len(transaction) for transaction in transactions]
plt.hist(transaction_sizes, bins=range(1, max(transaction_sizes)+2), alpha=0.7, color='brown')
plt.xlabel('Transaction Size')
plt.ylabel('Frequency')
plt.title('Transaction Size Distribution')
plt.xticks(range(1, max(transaction_sizes)+1))

plt.tight_layout()
plt.savefig('apriori_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Plot saved as 'apriori_analysis.png'")