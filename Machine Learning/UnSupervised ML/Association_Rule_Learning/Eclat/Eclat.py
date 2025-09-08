import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import networkx as nx
import seaborn as sns
from collections import defaultdict

# DATA LOAD
np.random.seed(42)

# Generate synthetic e-commerce transaction dataset
items = ['laptop', 'mouse', 'keyboard', 'monitor', 'headphones', 'webcam', 'speaker', 'printer',
         'tablet', 'smartphone', 'charger', 'case', 'cable', 'adapter', 'battery', 'memory_card',
         'hard_drive', 'software', 'antivirus', 'game']

transactions = []
transaction_ids = []

print("=== Eclat Algorithm for Frequent Itemset Mining ===")

# Generate 550 transactions with realistic e-commerce patterns
for i in range(550):
    transaction = []
    transaction_id = f'T{i+1:03d}'
    
    # Create realistic shopping patterns
    if np.random.random() < 0.4:  # Computer setup bundle
        if np.random.random() < 0.9:
            transaction.extend(['laptop', 'mouse'])
        if np.random.random() < 0.7:
            transaction.append('keyboard')
        if np.random.random() < 0.5:
            transaction.append('monitor')
        if np.random.random() < 0.3:
            transaction.append('headphones')
    
    if np.random.random() < 0.3:  # Mobile accessories
        if np.random.random() < 0.8:
            transaction.extend(['smartphone', 'charger'])
        if np.random.random() < 0.6:
            transaction.append('case')
        if np.random.random() < 0.4:
            transaction.append('headphones')
    
    if np.random.random() < 0.25:  # Gaming setup
        if np.random.random() < 0.8:
            transaction.extend(['laptop', 'mouse', 'headphones'])
        if np.random.random() < 0.5:
            transaction.append('game')
        if np.random.random() < 0.3:
            transaction.append('speaker')
    
    if np.random.random() < 0.2:  # Office setup
        if np.random.random() < 0.9:
            transaction.extend(['laptop', 'printer'])
        if np.random.random() < 0.6:
            transaction.extend(['keyboard', 'mouse'])
        if np.random.random() < 0.4:
            transaction.append('software')
    
    if np.random.random() < 0.15:  # Tech enthusiast
        if np.random.random() < 0.8:
            transaction.extend(['tablet', 'smartphone'])
        if np.random.random() < 0.6:
            transaction.extend(['hard_drive', 'memory_card'])
        if np.random.random() < 0.3:
            transaction.append('battery')
    
    # Add random items
    num_random = np.random.poisson(1.5)
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

# Convert to vertical format (TID-itemset representation)
vertical_data = defaultdict(list)
for tid, transaction in enumerate(transactions):
    for item in transaction:
        vertical_data[item].append(tid)

copyData = vertical_data.copy()

print(f"\nFirst few transactions:")
for i in range(3):
    print(f"{transaction_ids[i]}: {transactions[i]}")

print(f"\nVertical representation (sample):")
for item in list(items)[:3]:
    print(f"{item}: appears in transactions {sorted(vertical_data[item])[:10]}... (total: {len(vertical_data[item])})")

# ECLAT ALGORITHM IMPLEMENTATION
def calculate_support_eclat(tidset, total_transactions):
    """Calculate support for a TID set"""
    return len(tidset) / total_transactions

def intersect_tidsets(tidset1, tidset2):
    """Intersect two TID sets"""
    return list(set(tidset1) & set(tidset2))

def eclat_algorithm(vertical_data, min_support=0.05, total_transactions=None):
    """Implement Eclat algorithm for frequent itemset mining"""
    if total_transactions is None:
        total_transactions = len(transactions)
    
    frequent_itemsets = {}
    
    print(f"\n=== Eclat Algorithm (min_support={min_support}) ===")
    
    # Level 1: Individual items
    level_1 = {}
    for item, tidset in vertical_data.items():
        support = calculate_support_eclat(tidset, total_transactions)
        if support >= min_support:
            level_1[frozenset([item])] = {
                'tidset': tidset,
                'support': support
            }
    
    frequent_itemsets[1] = level_1
    print(f"Level 1: {len(level_1)} frequent items")
    
    # Generate higher level itemsets
    def eclat_recursive(current_level, level_num):
        if not current_level:
            return
        
        next_level = {}
        itemsets = list(current_level.keys())
        
        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                itemset1 = itemsets[i]
                itemset2 = itemsets[j]
                
                # Check if itemsets can be joined (differ by exactly one item)
                union = itemset1 | itemset2
                if len(union) == level_num + 1:
                    # Intersect TID sets
                    tidset1 = current_level[itemset1]['tidset']
                    tidset2 = current_level[itemset2]['tidset']
                    new_tidset = intersect_tidsets(tidset1, tidset2)
                    
                    if new_tidset:
                        support = calculate_support_eclat(new_tidset, total_transactions)
                        if support >= min_support:
                            next_level[union] = {
                                'tidset': new_tidset,
                                'support': support
                            }
        
        if next_level:
            frequent_itemsets[level_num + 1] = next_level
            print(f"Level {level_num + 1}: {len(next_level)} frequent itemsets")
            eclat_recursive(next_level, level_num + 1)
    
    eclat_recursive(level_1, 1)
    return frequent_itemsets

def generate_association_rules_eclat(frequent_itemsets, min_confidence=0.6):
    """Generate association rules from frequent itemsets found by Eclat"""
    rules = []
    
    print(f"\n=== Generating Association Rules (min_confidence={min_confidence}) ===")
    
    for k in range(2, len(frequent_itemsets) + 1):
        for itemset, data in frequent_itemsets[k].items():
            itemset_list = list(itemset)
            itemset_support = data['support']
            
            # Generate all possible antecedent-consequent pairs
            for i in range(1, len(itemset_list)):
                for antecedent in combinations(itemset_list, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    if antecedent in frequent_itemsets[len(antecedent)]:
                        antecedent_support = frequent_itemsets[len(antecedent)][antecedent]['support']
                        confidence = itemset_support / antecedent_support
                        
                        if confidence >= min_confidence:
                            # Calculate lift
                            consequent_support = 0
                            if len(consequent) == 1:
                                consequent_item = list(consequent)[0]
                                if frozenset([consequent_item]) in frequent_itemsets[1]:
                                    consequent_support = frequent_itemsets[1][frozenset([consequent_item])]['support']
                            
                            lift = confidence / consequent_support if consequent_support > 0 else float('inf')
                            
                            rules.append({
                                'antecedent': list(antecedent),
                                'consequent': list(consequent),
                                'support': itemset_support,
                                'confidence': confidence,
                                'lift': lift,
                                'antecedent_support': antecedent_support,
                                'consequent_support': consequent_support
                            })
    
    return sorted(rules, key=lambda x: x['confidence'], reverse=True)

# MODEL
min_support = 0.08
min_confidence = 0.6

frequent_itemsets = eclat_algorithm(vertical_data, min_support, len(transactions))
association_rules = generate_association_rules_eclat(frequent_itemsets, min_confidence)

# METRICS
print(f"\n=== Eclat Results ===")
total_frequent = sum(len(level) for level in frequent_itemsets.values())
print(f"Total frequent itemsets: {total_frequent}")
print(f"Total association rules: {len(association_rules)}")

# Show frequent itemsets by level
for level, itemsets in frequent_itemsets.items():
    print(f"Level {level}: {len(itemsets)} itemsets")
    if level <= 2:  # Show examples for smaller itemsets
        for itemset, data in list(itemsets.items())[:3]:
            print(f"  {set(itemset)}: support={data['support']:.4f}")

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
item_support = {item: calculate_support_eclat(tidset, len(transactions)) for item, tidset in vertical_data.items()}
sorted_items = sorted(item_support.items(), key=lambda x: x[1], reverse=True)
item_names, supports = zip(*sorted_items[:15])

plt.barh(range(len(item_names)), supports, color='lightblue', alpha=0.7)
plt.yticks(range(len(item_names)), item_names)
plt.xlabel('Support')
plt.title('Top 15 Items by Support')
plt.gca().invert_yaxis()

# Plot 2: Frequent itemsets by level
plt.subplot(3, 3, 2)
levels = list(frequent_itemsets.keys())
counts = [len(frequent_itemsets[level]) for level in levels]
plt.bar(levels, counts, color='orange', alpha=0.7)
plt.xlabel('Itemset Size')
plt.ylabel('Number of Frequent Itemsets')
plt.title('Frequent Itemsets by Size (Eclat)')

# Plot 3: Support vs Confidence scatter
plt.subplot(3, 3, 3)
if association_rules:
    supports = [rule['support'] for rule in association_rules]
    confidences = [rule['confidence'] for rule in association_rules]
    lifts = [rule['lift'] for rule in association_rules if rule['lift'] != float('inf')]
    
    if lifts:
        scatter = plt.scatter(supports[:len(lifts)], confidences[:len(lifts)], c=lifts, cmap='viridis', alpha=0.7)
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Association Rules: Support vs Confidence')
        plt.colorbar(scatter, label='Lift')

# Plot 4: Top rules by confidence
plt.subplot(3, 3, 4)
if len(association_rules) >= 8:
    top_rules = association_rules[:8]
    rule_labels = [f"{', '.join(rule['antecedent'][:2])}→{', '.join(rule['consequent'][:2])}" for rule in top_rules]
    confidences = [rule['confidence'] for rule in top_rules]
    
    plt.barh(range(len(rule_labels)), confidences, color='green', alpha=0.7)
    plt.yticks(range(len(rule_labels)), rule_labels, fontsize=8)
    plt.xlabel('Confidence')
    plt.title('Top 8 Rules by Confidence')
    plt.gca().invert_yaxis()

# Plot 5: Network graph of associations
plt.subplot(3, 3, 5)
if len(association_rules) >= 6:
    G = nx.DiGraph()
    top_rules_graph = association_rules[:6]  # Top 6 for better visualization
    
    for rule in top_rules_graph:
        antecedent_str = ', '.join(rule['antecedent'])
        consequent_str = ', '.join(rule['consequent'])
        G.add_edge(antecedent_str, consequent_str, weight=rule['confidence'])
    
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw(G, pos, with_labels=True, node_color='lightcoral', 
            node_size=1000, font_size=8, font_weight='bold',
            edge_color='gray', arrows=True, arrowsize=20)
    plt.title('Association Rules Network (Eclat)')

# Plot 6: Lift distribution
plt.subplot(3, 3, 6)
if association_rules:
    lifts = [rule['lift'] for rule in association_rules if rule['lift'] != float('inf')]
    if lifts:
        plt.hist(lifts, bins=15, alpha=0.7, color='purple')
        plt.axvline(x=1, color='red', linestyle='--', label='Lift = 1 (Independence)')
        plt.xlabel('Lift')
        plt.ylabel('Frequency')
        plt.title('Distribution of Lift Values')
        plt.legend()

# CUSTOM INPUT
print("\n=== Custom E-commerce Basket Analysis ===")
print("Enter items in your shopping cart (comma-separated):")
user_basket_input = input("Your cart: ").strip()
user_basket = [item.strip().lower() for item in user_basket_input.split(',') if item.strip()]

print(f"\nYour cart: {user_basket}")

# Find applicable rules
applicable_rules = []
for rule in association_rules:
    antecedent_lower = [item.lower() for item in rule['antecedent']]
    if all(item in user_basket for item in antecedent_lower):
        applicable_rules.append(rule)

if applicable_rules:
    print(f"\n=== Product Recommendations based on Eclat Rules ===")
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
        print(f"\n=== Top Recommended Products ===")
        for item, total_confidence in sorted_recommendations[:5]:
            print(f"• {item} (total confidence: {total_confidence:.2%})")
else:
    print("No specific recommendations found for your cart.")
    print("Consider popular items:", [item for item, _ in sorted_items[:5]])

# Plot 7: User cart visualization
plt.subplot(3, 3, 7)
if user_basket and association_rules:
    # Show recommendations
    applicable_confidences = [rule['confidence'] for rule in applicable_rules[:8]]
    applicable_labels = [f"{', '.join(rule['consequent'])}" for rule in applicable_rules[:8]]
    
    if applicable_confidences:
        plt.barh(range(len(applicable_labels)), applicable_confidences, color='red', alpha=0.7)
        plt.yticks(range(len(applicable_labels)), applicable_labels, fontsize=8)
        plt.xlabel('Confidence')
        plt.title('Recommendations for Your Cart')
        plt.gca().invert_yaxis()
    else:
        plt.text(0.5, 0.5, 'No specific\nrecommendations\nfound', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title('Recommendations for Your Cart')

# Plot 8: Algorithm comparison placeholder
plt.subplot(3, 3, 8)
# Placeholder for algorithm comparison (Eclat vs Apriori)
algorithms = ['Apriori', 'Eclat']
efficiency = [0.7, 0.9]  # Eclat is generally more efficient
plt.bar(algorithms, efficiency, color=['blue', 'orange'], alpha=0.7)
plt.ylabel('Efficiency Score')
plt.title('Algorithm Efficiency Comparison')
plt.ylim(0, 1)

# Plot 9: Transaction size distribution
plt.subplot(3, 3, 9)
transaction_sizes = [len(transaction) for transaction in transactions]
plt.hist(transaction_sizes, bins=range(1, max(transaction_sizes)+2), alpha=0.7, color='brown')
plt.xlabel('Transaction Size')
plt.ylabel('Frequency')
plt.title('Transaction Size Distribution')
plt.xticks(range(1, max(transaction_sizes)+1))

plt.tight_layout()
plt.savefig('eclat_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Plot saved as 'eclat_analysis.png'")