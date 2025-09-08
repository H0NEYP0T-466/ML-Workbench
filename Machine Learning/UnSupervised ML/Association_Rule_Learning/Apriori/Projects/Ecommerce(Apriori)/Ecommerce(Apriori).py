import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import seaborn as sns

# DATA
np.random.seed(42)

# Generate synthetic e-commerce dataset with categories
categories = {
    'fashion': ['t_shirt', 'jeans', 'dress', 'shoes', 'jacket', 'hat', 'belt', 'handbag', 'watch', 'sunglasses'],
    'electronics': ['smartphone', 'laptop', 'headphones', 'tablet', 'camera', 'smartwatch', 'speaker', 'charger', 'keyboard', 'mouse'],
    'home': ['pillow', 'blanket', 'lamp', 'candle', 'picture_frame', 'vase', 'cushion', 'rug', 'mirror', 'plant_pot'],
    'kitchen': ['coffee_maker', 'blender', 'toaster', 'pan', 'knife_set', 'cutting_board', 'measuring_cups', 'mixer', 'pot', 'spatula'],
    'beauty': ['moisturizer', 'shampoo', 'lipstick', 'foundation', 'mascara', 'nail_polish', 'perfume', 'face_mask', 'soap', 'lotion'],
    'sports': ['running_shoes', 'yoga_mat', 'dumbbells', 'water_bottle', 'fitness_tracker', 'resistance_bands', 'tennis_ball', 'basketball', 'swimsuit', 'bike_helmet'],
    'books': ['novel', 'cookbook', 'self_help', 'biography', 'textbook', 'magazine', 'comic_book', 'poetry', 'travel_guide', 'history_book'],
    'toys': ['action_figure', 'board_game', 'puzzle', 'doll', 'building_blocks', 'stuffed_animal', 'toy_car', 'coloring_book', 'card_game', 'educational_toy']
}

# Flatten all items
all_items = []
item_categories = {}
for category, items in categories.items():
    all_items.extend(items)
    for item in items:
        item_categories[item] = category

transactions = []
customer_ids = []

print("=== Apriori E-commerce Analysis ===")

# Generate 750 realistic e-commerce transactions
for i in range(750):
    transaction = []
    customer_id = f'CUSTOMER_{i+1:03d}'
    
    # Different shopping patterns based on primary category interest
    primary_category = np.random.choice(list(categories.keys()), 
                                      p=[0.18, 0.16, 0.14, 0.12, 0.12, 0.10, 0.10, 0.08])
    
    # Add 2-4 items from primary category
    primary_items = np.random.choice(categories[primary_category], 
                                   size=np.random.randint(2, 5), replace=False)
    transaction.extend(primary_items)
    
    # Cross-category patterns
    if primary_category == 'fashion':
        # Fashion buyers often buy beauty products
        if np.random.random() < 0.6:
            transaction.extend(np.random.choice(categories['beauty'], size=1))
        # Sometimes home accessories
        if np.random.random() < 0.3:
            transaction.extend(np.random.choice(categories['home'], size=1))
    
    elif primary_category == 'electronics':
        # Electronics buyers often buy accessories
        if np.random.random() < 0.5:
            related_electronics = ['charger', 'headphones', 'keyboard', 'mouse']
            available_related = [item for item in related_electronics if item not in transaction]
            if available_related:
                transaction.extend(np.random.choice(available_related, size=1))
        # Sometimes books (tech books)
        if np.random.random() < 0.3:
            transaction.extend(np.random.choice(categories['books'], size=1))
    
    elif primary_category == 'home':
        # Home buyers often buy kitchen items
        if np.random.random() < 0.7:
            transaction.extend(np.random.choice(categories['kitchen'], size=1))
        # Sometimes beauty (bathroom items)
        if np.random.random() < 0.4:
            transaction.extend(np.random.choice(categories['beauty'], size=1))
    
    elif primary_category == 'kitchen':
        # Kitchen buyers often buy home items
        if np.random.random() < 0.6:
            transaction.extend(np.random.choice(categories['home'], size=1))
        # Sometimes books (cookbooks)
        if np.random.random() < 0.4:
            cookbooks = ['cookbook']
            if 'cookbook' not in transaction:
                transaction.append('cookbook')
    
    elif primary_category == 'beauty':
        # Beauty buyers often buy fashion
        if np.random.random() < 0.5:
            transaction.extend(np.random.choice(categories['fashion'], size=1))
        # Sometimes home (mirrors, etc.)
        if np.random.random() < 0.3:
            transaction.extend(np.random.choice(categories['home'], size=1))
    
    elif primary_category == 'sports':
        # Sports buyers often buy electronics (fitness trackers)
        if np.random.random() < 0.5:
            fitness_electronics = ['fitness_tracker', 'smartwatch', 'headphones']
            available_fitness = [item for item in fitness_electronics if item not in transaction and item in categories['electronics']]
            if available_fitness:
                transaction.extend(np.random.choice(available_fitness, size=1))
            elif 'fitness_tracker' not in transaction:
                transaction.append('fitness_tracker')
    
    elif primary_category == 'books':
        # Book buyers often buy home items (reading accessories)
        if np.random.random() < 0.4:
            reading_items = ['lamp', 'pillow', 'blanket']
            available_reading = [item for item in reading_items if item not in transaction]
            if available_reading:
                transaction.extend(np.random.choice(available_reading, size=1))
    
    elif primary_category == 'toys':
        # Toy buyers (parents) often buy books for kids
        if np.random.random() < 0.6:
            kid_books = ['coloring_book', 'educational_toy']
            available_kid_items = [item for item in kid_books if item not in transaction]
            if available_kid_items:
                transaction.extend(np.random.choice(available_kid_items, size=1))
        # Sometimes home items (for family)
        if np.random.random() < 0.3:
            transaction.extend(np.random.choice(categories['home'], size=1))
    
    # Add some random cross-category items (impulse purchases)
    if np.random.random() < 0.3:
        random_category = np.random.choice([cat for cat in categories.keys() if cat != primary_category])
        random_item = np.random.choice(categories[random_category], size=1)
        transaction.extend(random_item)
    
    # Specific item correlations
    if 'smartphone' in transaction and np.random.random() < 0.7:
        if 'charger' not in transaction:
            transaction.append('charger')
    if 'laptop' in transaction and np.random.random() < 0.6:
        if 'mouse' not in transaction:
            transaction.append('mouse')
    if 'coffee_maker' in transaction and np.random.random() < 0.5:
        if 'measuring_cups' not in transaction:
            transaction.append('measuring_cups')
    if 'yoga_mat' in transaction and np.random.random() < 0.4:
        if 'water_bottle' not in transaction:
            transaction.append('water_bottle')
    
    # Remove duplicates and ensure minimum transaction size
    transaction = list(set(transaction))
    if len(transaction) < 2:
        additional_items = np.random.choice(all_items, size=2, replace=False)
        transaction.extend(additional_items)
        transaction = list(set(transaction))
    
    transactions.append(transaction)
    customer_ids.append(customer_id)

print(f"Generated {len(transactions)} e-commerce transactions")
print(f"Average cart size: {np.mean([len(t) for t in transactions]):.2f}")
print(f"Unique products: {len(set(item for transaction in transactions for item in transaction))}")

# Convert to binary matrix format for Apriori
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
copyData = df.copy()

# Save dataset to CSV
transactions_df = pd.DataFrame({'Customer_ID': customer_ids, 'Items': [','.join(t) for t in transactions]})
transactions_df.to_csv('ecommerce_dataset.csv', index=False)

print(f"\nDataset shape: {copyData.shape}")
print(f"Sample e-commerce carts:")
for i in range(3):
    items_in_cart = [item for item in te.columns_ if df.iloc[i][item]]
    cart_categories = list(set([item_categories.get(item, 'unknown') for item in items_in_cart]))
    print(f"Cart {i+1}: {items_in_cart} (Categories: {cart_categories})")

# MODEL
print(f"\n=== Apriori E-commerce Cross-selling Analysis ===")

# Parameters optimized for e-commerce data
min_support = 0.03  # 3% minimum support 
min_confidence = 0.4  # 40% minimum confidence  
min_lift = 1.1  # Lift > 1.1 indicates positive association

# Generate frequent itemsets using Apriori
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, verbose=1, max_len=4)
print(f"Found {len(frequent_itemsets)} frequent itemsets with min_support={min_support}")

# RULES
# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))
rules_filtered = rules[rules['lift'] >= min_lift].sort_values('lift', ascending=False).reset_index(drop=True)

print(f"Generated {len(rules_filtered)} cross-selling rules with min_confidence={min_confidence} and min_lift={min_lift}")

# Print top association rules sorted by lift
print(f"\n=== Top 15 Cross-selling Rules (Sorted by Lift) ===")
print(f"{'#':>3} {'If Customer Buys':>30} {'Then Recommend':>25} {'Support':>8} {'Confidence':>10} {'Lift':>8}")
print("-" * 95)
for i, rule in rules_filtered.head(15).iterrows():
    antecedent = ', '.join(list(rule['antecedents']))[:28]
    consequent = ', '.join(list(rule['consequents']))[:23]
    print(f"{i+1:>3} {antecedent:>30} {consequent:>25} {rule['support']:>8.4f} {rule['confidence']:>10.4f} {rule['lift']:>8.4f}")

# PLOTS
plt.figure(figsize=(15, 10))

# Plot 1: Heatmap of support vs confidence
plt.subplot(2, 3, 1)
if len(rules_filtered) > 0:
    # Create bins for support and confidence
    support_bins = np.linspace(rules_filtered['support'].min(), rules_filtered['support'].max(), 8)
    confidence_bins = np.linspace(rules_filtered['confidence'].min(), rules_filtered['confidence'].max(), 8)
    
    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(rules_filtered['support'], rules_filtered['confidence'], 
                                         bins=[support_bins, confidence_bins])
    
    # Create heatmap
    sns.heatmap(hist.T, annot=True, fmt='.0f', cmap='YlOrRd', 
                xticklabels=[f'{x:.3f}' for x in support_bins[:-1]], 
                yticklabels=[f'{y:.2f}' for y in confidence_bins[:-1]])
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Heatmap: Support vs Confidence')

# Plot 2: Top 10 rules bar chart
plt.subplot(2, 3, 2)
if len(rules_filtered) >= 10:
    top_10_rules = rules_filtered.head(10)
    rule_labels = []
    for _, rule in top_10_rules.iterrows():
        ant = ', '.join(list(rule['antecedents']))[:12]
        cons = ', '.join(list(rule['consequents']))[:12]
        rule_labels.append(f"{ant}→{cons}")
    
    plt.barh(range(len(rule_labels)), top_10_rules['lift'], 
             color='lightsteelblue', alpha=0.8)
    plt.yticks(range(len(rule_labels)), rule_labels, fontsize=8)
    plt.xlabel('Lift')
    plt.title('Top 10 Cross-selling Rules by Lift')
    plt.gca().invert_yaxis()

# Plot 3: Category distribution
plt.subplot(2, 3, 3)
category_counts = {}
for transaction in transactions:
    transaction_categories = set()
    for item in transaction:
        if item in item_categories:
            transaction_categories.add(item_categories[item])
    for category in transaction_categories:
        category_counts[category] = category_counts.get(category, 0) + 1

categories_list = list(category_counts.keys())
counts = list(category_counts.values())
colors = plt.cm.Set3(np.linspace(0, 1, len(categories_list)))

plt.pie(counts, labels=categories_list, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Category Distribution in Transactions')

# Plot 4: Cross-category rules network
plt.subplot(2, 3, 4)
if len(rules_filtered) >= 8:
    G = nx.DiGraph()
    top_rules = rules_filtered.head(8)
    
    for _, rule in top_rules.iterrows():
        # Get categories for antecedent and consequent
        ant_items = list(rule['antecedents'])
        cons_items = list(rule['consequents'])
        
        ant_categories = set()
        cons_categories = set()
        
        for item in ant_items:
            if item in item_categories:
                ant_categories.add(item_categories[item])
        for item in cons_items:
            if item in item_categories:
                cons_categories.add(item_categories[item])
        
        for ant_cat in ant_categories:
            for cons_cat in cons_categories:
                if ant_cat != cons_cat:
                    if G.has_edge(ant_cat, cons_cat):
                        G[ant_cat][cons_cat]['weight'] += rule['lift']
                    else:
                        G.add_edge(ant_cat, cons_cat, weight=rule['lift'])
    
    if G.number_of_edges() > 0:
        pos = nx.spring_layout(G, k=3, iterations=50)
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1200, font_size=9, font_weight='bold',
                edge_color='gray', arrows=True, arrowsize=20,
                width=[w/max(edge_weights)*3 for w in edge_weights])
        plt.title('Cross-Category Recommendations')

# Plot 5: Support vs Lift scatter
plt.subplot(2, 3, 5)
if len(rules_filtered) > 0:
    scatter = plt.scatter(rules_filtered['support'], rules_filtered['lift'], 
                         c=rules_filtered['confidence'], cmap='plasma', alpha=0.7, s=60)
    plt.xlabel('Support')
    plt.ylabel('Lift')
    plt.title('Rules: Support vs Lift')
    plt.colorbar(scatter, label='Confidence')

# Plot 6: Transaction size by primary category
plt.subplot(2, 3, 6)
category_sizes = {cat: [] for cat in categories.keys()}
for i, transaction in enumerate(transactions):
    transaction_categories = []
    for item in transaction:
        if item in item_categories:
            transaction_categories.append(item_categories[item])
    
    if transaction_categories:
        primary_cat = max(set(transaction_categories), key=transaction_categories.count)
        category_sizes[primary_cat].append(len(transaction))

# Create box plot
category_data = []
category_labels = []
for cat, sizes in category_sizes.items():
    if sizes:  # Only include categories with data
        category_data.append(sizes)
        category_labels.append(cat)

if category_data:
    plt.boxplot(category_data, tick_labels=category_labels)
    plt.xticks(rotation=45)
    plt.ylabel('Transaction Size')
    plt.title('Cart Size by Primary Category')

plt.tight_layout()
plt.savefig('ecommerce_apriori_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# CUSTOM INPUT
print("\n=== Your Shopping Cart Recommendations ===")
try:
    user_input = input("Enter items in your cart (comma-separated): ").strip()
    if user_input and user_input.lower() not in ['skip', 'default', '']:
        user_items = [item.strip().lower().replace(' ', '_') for item in user_input.split(',')]
        user_items = [item for item in user_items if item in te.columns_]
        
        if user_items:
            print(f"\nYour shopping cart: {[item.replace('_', ' ').title() for item in user_items]}")
            
            # Get user's categories
            user_categories = set()
            for item in user_items:
                if item in item_categories:
                    user_categories.add(item_categories[item])
            print(f"Your primary categories: {list(user_categories)}")
            
            # Find cross-selling recommendations
            recommendations = []
            total_potential = 0
            
            for _, rule in rules_filtered.iterrows():
                antecedent_items = set(rule['antecedents'])
                consequent_items = set(rule['consequents'])
                user_items_set = set(user_items)
                
                # If user has all antecedent items, recommend consequent items
                if antecedent_items.issubset(user_items_set):
                    for item in consequent_items:
                        if item not in user_items_set:
                            item_category = item_categories.get(item, 'unknown')
                            recommendations.append({
                                'item': item,
                                'category': item_category,
                                'confidence': rule['confidence'],
                                'lift': rule['lift'],
                                'support': rule['support'],
                                'revenue_potential': rule['confidence'] * rule['lift'] * 20  # Scoring metric
                            })
            
            if recommendations:
                # Sort by revenue potential and remove duplicates
                seen_items = set()
                print(f"\n=== Cross-selling Recommendations ===")
                for i, rec in enumerate(sorted(recommendations, key=lambda x: x['revenue_potential'], reverse=True)[:8]):
                    if rec['item'] not in seen_items:
                        print(f"{len(seen_items)+1}. {rec['item'].replace('_', ' ').title()} (Category: {rec['category'].title()})")
                        print(f"   Confidence: {rec['confidence']:.1%}, Lift: {rec['lift']:.2f}")
                        print(f"   Revenue potential: {rec['revenue_potential']:.0f}/100")
                        total_potential += rec['revenue_potential']
                        seen_items.add(rec['item'])
                
                print(f"\nTotal cross-selling potential score: {total_potential:.0f}")
            else:
                print("No specific cross-selling opportunities found.")
                # Show some popular product combinations
                print("\nPopular product bundles include:")
                for _, rule in rules_filtered.head(3).iterrows():
                    items = list(rule['antecedents']) + list(rule['consequents'])
                    bundle_name = ' + '.join([item.replace('_', ' ').title() for item in items])
                    print(f"- {bundle_name} (Lift: {rule['lift']:.2f})")
        else:
            print("No valid products found in your input.")
    else:
        user_categories = ['electronics', 'fashion']  # Default for demonstration
        print(f"Using default categories: {user_categories}")
        
except (KeyboardInterrupt, EOFError):
    user_categories = ['electronics']  # Default when skipped
    print(f"\nUsing default category: {user_categories}")

print("\nAnalysis complete! Plots saved as 'ecommerce_apriori_analysis.png'")