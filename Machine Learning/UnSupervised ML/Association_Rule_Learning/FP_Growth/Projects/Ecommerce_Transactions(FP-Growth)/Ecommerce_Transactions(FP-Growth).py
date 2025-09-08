import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import seaborn as sns

# DATA LOAD
np.random.seed(42)

# Generate synthetic e-commerce transaction dataset
ecommerce_items = [
    # Electronics
    'smartphone', 'laptop', 'tablet', 'headphones', 'smartwatch', 'camera', 'speaker', 'charger', 'cable', 'mouse',
    'keyboard', 'monitor', 'hard_drive', 'memory_card', 'power_bank', 'router', 'webcam',
    # Fashion
    'jeans', 't_shirt', 'dress', 'shoes', 'jacket', 'hat', 'bag', 'sunglasses', 'belt', 'scarf',
    'watch', 'jewelry', 'socks', 'underwear',
    # Home & Garden
    'coffee_maker', 'blender', 'vacuum', 'lamp', 'pillow', 'blanket', 'curtains', 'plant', 'pot',
    'kitchen_knife', 'cutting_board', 'cookware',
    # Books & Media
    'book', 'ebook', 'audiobook', 'magazine', 'movie', 'music', 'game', 'toy',
    # Sports & Outdoor
    'fitness_tracker', 'yoga_mat', 'dumbbells', 'water_bottle', 'backpack', 'tent', 'bike_helmet',
    # Beauty & Health
    'shampoo', 'skincare', 'makeup', 'supplement', 'toothbrush', 'soap'
]

transactions = []
transaction_ids = []
customer_ids = []

print("=== FP-Growth for E-commerce Transaction Analysis ===")

# Generate 750 realistic e-commerce transactions
for i in range(750):
    transaction = []
    transaction_id = f'ORDER_{i+1:04d}'
    customer_id = f'CUST_{np.random.randint(1, 300):03d}'  # Some customers make multiple orders
    
    # Different shopping patterns based on customer behavior
    purchase_type = np.random.choice(['tech_enthusiast', 'fashion_shopper', 'home_improvement', 
                                    'student', 'fitness_fan', 'casual_browser'], 
                                   p=[0.2, 0.25, 0.15, 0.15, 0.1, 0.15])
    
    if purchase_type == 'tech_enthusiast':
        # Tech enthusiast: electronics, accessories
        if np.random.random() < 0.8:  # Main device
            main_device = np.random.choice(['smartphone', 'laptop', 'tablet', 'camera'])
            transaction.append(main_device)
        if np.random.random() < 0.7:  # Accessories
            if 'smartphone' in transaction:
                transaction.extend(['charger', 'headphones', 'phone_case'])
            if 'laptop' in transaction:
                transaction.extend(['mouse', 'laptop_bag', 'external_drive'])
        if np.random.random() < 0.6:  # Additional tech
            transaction.extend(['power_bank', 'cable'])
        if np.random.random() < 0.4:  # Smart home
            transaction.extend(['smart_speaker', 'smart_bulb'])
    
    elif purchase_type == 'fashion_shopper':
        # Fashion shopper: clothing, accessories
        if np.random.random() < 0.9:  # Clothing essentials
            transaction.extend(['jeans', 't_shirt'])
        if np.random.random() < 0.7:  # Footwear
            transaction.append('shoes')
        if np.random.random() < 0.6:  # Accessories
            transaction.extend(['bag', 'sunglasses'])
        if np.random.random() < 0.5:  # Seasonal items
            season_items = np.random.choice(['jacket', 'dress', 'hat'], size=2, replace=False)
            transaction.extend(season_items)
        if np.random.random() < 0.3:  # Jewelry
            transaction.extend(['watch', 'jewelry'])
    
    elif purchase_type == 'home_improvement':
        # Home improvement: appliances, decor
        if np.random.random() < 0.8:  # Kitchen appliances
            transaction.extend(['coffee_maker', 'blender'])
        if np.random.random() < 0.6:  # Home decor
            transaction.extend(['lamp', 'pillow', 'plant'])
        if np.random.random() < 0.5:  # Cleaning
            transaction.append('vacuum')
        if np.random.random() < 0.4:  # Kitchen tools
            transaction.extend(['kitchen_knife', 'cutting_board'])
    
    elif purchase_type == 'student':
        # Student: books, basic electronics, budget items
        if np.random.random() < 0.8:  # Study materials
            transaction.extend(['book', 'laptop'])
        if np.random.random() < 0.6:  # Basic accessories
            transaction.extend(['headphones', 'backpack'])
        if np.random.random() < 0.4:  # Entertainment
            transaction.extend(['game', 'movie'])
        if np.random.random() < 0.3:  # Budget items
            transaction.append('water_bottle')
    
    elif purchase_type == 'fitness_fan':
        # Fitness enthusiast: sports equipment, health items
        if np.random.random() < 0.9:  # Fitness gear
            transaction.extend(['fitness_tracker', 'yoga_mat'])
        if np.random.random() < 0.7:  # Equipment
            transaction.extend(['dumbbells', 'water_bottle'])
        if np.random.random() < 0.5:  # Health supplements
            transaction.append('supplement')
        if np.random.random() < 0.4:  # Outdoor gear
            transaction.extend(['backpack', 'bike_helmet'])
    
    else:  # casual_browser
        # Casual browser: random mix of items
        category_items = {
            'electronics': ['smartphone', 'headphones', 'charger'],
            'fashion': ['t_shirt', 'jeans', 'shoes'],
            'home': ['coffee_maker', 'pillow'],
            'books': ['book', 'movie'],
            'beauty': ['shampoo', 'skincare']
        }
        
        selected_category = np.random.choice(list(category_items.keys()))
        num_items = np.random.randint(1, 4)
        selected_items = np.random.choice(category_items[selected_category], 
                                        size=min(num_items, len(category_items[selected_category])), 
                                        replace=False)
        transaction.extend(selected_items)
    
    # Add some random cross-category items (cross-selling effect)
    if np.random.random() < 0.3:
        num_random = np.random.randint(1, 3)
        random_items = np.random.choice(ecommerce_items, size=num_random, replace=False)
        transaction.extend(random_items)
    
    # Remove duplicates and filter empty
    transaction = list(set(transaction))
    if len(transaction) > 0:
        transactions.append(transaction)
        transaction_ids.append(transaction_id)
        customer_ids.append(customer_id)

print(f"Generated {len(transactions)} e-commerce transactions")
print(f"Average order size: {np.mean([len(t) for t in transactions]):.2f} items")
print(f"Unique products: {len(set(item for transaction in transactions for item in transaction))}")
print(f"Unique customers: {len(set(customer_ids))}")

# Convert to binary matrix format for FP-Growth
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
copyData = df.copy()

print(f"\nDataset shape: {copyData.shape}")
print(f"Sample e-commerce orders:")
for i in range(3):
    items_in_order = [item for item in te.columns_ if df.iloc[i][item]]
    print(f"Order {i+1}: {items_in_order}")

# MODEL
print(f"\n=== FP-Growth E-commerce Analysis ===")

# Optimized parameters for e-commerce data
min_support = 0.04  # 4% minimum support
min_confidence = 0.5  # 50% minimum confidence
min_lift = 1.3  # Lift > 1.3 indicates strong cross-selling opportunity

# Generate frequent itemsets using FP-Growth
frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True, verbose=1)
print(f"Found {len(frequent_itemsets)} frequent itemsets with min_support={min_support}")

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))

# Filter rules by lift
rules_filtered = rules[rules['lift'] >= min_lift].copy()
rules_filtered = rules_filtered.sort_values('lift', ascending=False).reset_index(drop=True)

print(f"Generated {len(rules_filtered)} cross-selling rules with min_confidence={min_confidence} and min_lift={min_lift}")

# METRICS
print(f"\n=== E-commerce Cross-selling Analysis ===")
print(f"Total frequent itemsets: {len(frequent_itemsets)}")
print(f"Total cross-selling rules: {len(rules_filtered)}")
print(f"Average support: {rules_filtered['support'].mean():.4f}")
print(f"Average confidence: {rules_filtered['confidence'].mean():.4f}")
print(f"Average lift: {rules_filtered['lift'].mean():.4f}")

print(f"\n=== Top 15 Cross-selling Opportunities ===")
print(f"{'#':>3} {'When Customer Buys':>30} {'Recommend':>25} {'Support':>8} {'Confidence':>10} {'Lift':>8}")
print("-" * 95)
for i, rule in rules_filtered.head(15).iterrows():
    antecedent = ', '.join(list(rule['antecedents']))[:28]
    consequent = ', '.join(list(rule['consequents']))[:23]
    print(f"{i+1:>3} {antecedent:>30} {consequent:>25} {rule['support']:>8.4f} {rule['confidence']:>10.4f} {rule['lift']:>8.4f}")

# CUSTOM INPUT
print("\n=== Your Shopping Cart Recommendations ===")
try:
    user_input = input("Enter items in your cart (comma-separated): ").strip()
    if user_input and user_input.lower() not in ['skip', 'default', '']:
        user_items = [item.strip().lower().replace(' ', '_') for item in user_input.split(',')]
        user_items = [item for item in user_items if item in te.columns_]
        
        if user_items:
            print(f"\nYour shopping cart: {[item.replace('_', ' ').title() for item in user_items]}")
            
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
                                'based_on': list(antecedent_items),
                                'revenue_potential': rule['confidence'] * rule['lift'] * 100  # Simulated revenue score
                            })
            
            # Sort recommendations by revenue potential
            recommendations = sorted(recommendations, 
                                   key=lambda x: x['revenue_potential'], 
                                   reverse=True)
            
            print(f"\n=== Cross-selling Recommendations ===")
            if recommendations:
                seen_items = set()
                total_potential = 0
                for i, rec in enumerate(recommendations[:8]):
                    if rec['item'] not in seen_items:
                        item_name = rec['item'].replace('_', ' ').title()
                        based_on = [item.replace('_', ' ').title() for item in rec['based_on']]
                        print(f"{i+1}. {item_name}")
                        print(f"   Frequently bought with: {', '.join(based_on)}")
                        print(f"   Cross-sell probability: {rec['confidence']:.1%}")
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
        user_items = ['smartphone', 'headphones']  # Default for visualization
        print(f"Using default cart: {[item.replace('_', ' ').title() for item in user_items]}")
        
except (KeyboardInterrupt, EOFError):
    user_items = ['smartphone', 'headphones']  # Default for visualization
    print(f"\nUsing default cart: {[item.replace('_', ' ').title() for item in user_items]}")

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

plt.barh(range(len(display_names)), supports, color='lightcoral', alpha=0.7)
plt.yticks(range(len(display_names)), display_names)
plt.xlabel('Purchase Frequency')
plt.title('Top 15 Products by Sales')
plt.gca().invert_yaxis()

# Plot 2: Cross-selling rules analysis
plt.subplot(3, 3, 2)
if len(rules_filtered) > 0:
    scatter = plt.scatter(rules_filtered['support'], rules_filtered['confidence'], 
                         c=rules_filtered['lift'], cmap='viridis', alpha=0.7, s=60)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Cross-selling Rules Analysis')
    plt.colorbar(scatter, label='Lift (Cross-sell Strength)')

# Plot 3: Product categories sales distribution
plt.subplot(3, 3, 3)
categories = {
    'Electronics': ['smartphone', 'laptop', 'tablet', 'headphones', 'camera', 'charger'],
    'Fashion': ['jeans', 't_shirt', 'dress', 'shoes', 'bag', 'sunglasses'],
    'Home': ['coffee_maker', 'blender', 'vacuum', 'lamp', 'pillow'],
    'Books/Media': ['book', 'movie', 'music', 'game'],
    'Sports': ['fitness_tracker', 'yoga_mat', 'dumbbells', 'water_bottle']
}

category_sales = {}
for category, items in categories.items():
    category_sales[category] = sum(df[item].sum() for item in items if item in df.columns)

plt.pie(category_sales.values(), labels=category_sales.keys(), autopct='%1.1f%%', 
        colors=['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lavender'])
plt.title('Sales Distribution by Category')

# Plot 4: Order size distribution
plt.subplot(3, 3, 4)
order_sizes = [len(transaction) for transaction in transactions]
plt.hist(order_sizes, bins=range(1, max(order_sizes)+2), 
         alpha=0.7, color='mediumpurple', edgecolor='black')
plt.xlabel('Order Size (Number of Items)')
plt.ylabel('Number of Orders')
plt.title('E-commerce Order Size Distribution')

# Plot 5: Product recommendation network
plt.subplot(3, 3, 5)
if len(rules_filtered) >= 5:
    G = nx.DiGraph()
    top_rules_graph = rules_filtered.head(8)  # Top 8 for better visualization
    
    for _, rule in top_rules_graph.iterrows():
        for ant in rule['antecedents']:
            for cons in rule['consequents']:
                # Simplify names for better display
                ant_short = ant.replace('_', ' ')[:10]
                cons_short = cons.replace('_', ' ')[:10]
                G.add_edge(ant_short, cons_short, weight=rule['lift'])
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    nx.draw(G, pos, with_labels=True, node_color='lightsteelblue', 
            node_size=1000, font_size=8, font_weight='bold',
            edge_color='gray', arrows=True, arrowsize=20)
    plt.title('Cross-selling Network')

# Plot 6: Revenue potential analysis
plt.subplot(3, 3, 6)
if len(rules_filtered) >= 10:
    rules_filtered['revenue_score'] = rules_filtered['confidence'] * rules_filtered['lift'] * rules_filtered['support']
    top_revenue = rules_filtered.nlargest(10, 'revenue_score')
    
    rule_labels = []
    for _, rule in top_revenue.iterrows():
        ant = ', '.join([item.replace('_', ' ')[:8] for item in rule['antecedents']])
        cons = ', '.join([item.replace('_', ' ')[:8] for item in rule['consequents']])
        rule_labels.append(f"{ant} → {cons}")
    
    plt.barh(range(len(rule_labels)), top_revenue['revenue_score'], color='gold', alpha=0.7)
    plt.yticks(range(len(rule_labels)), rule_labels)
    plt.xlabel('Revenue Potential Score')
    plt.title('Top Cross-selling Opportunities')
    plt.gca().invert_yaxis()

# Plot 7: Customer behavior analysis
plt.subplot(3, 3, 7)
# Simulate customer segmentation based on order patterns
purchase_patterns = ['Tech Enthusiast', 'Fashion Shopper', 'Home Improver', 'Student', 'Fitness Fan', 'Casual Browser']
pattern_counts = [150, 188, 113, 113, 75, 113]  # Based on our probabilities
colors = ['steelblue', 'pink', 'lightgreen', 'orange', 'red', 'gray']

plt.bar(purchase_patterns, pattern_counts, color=colors, alpha=0.7)
plt.xlabel('Customer Type')
plt.ylabel('Number of Orders')
plt.title('Orders by Customer Behavior')
plt.xticks(rotation=45)

# Plot 8: Cross-selling success metrics
plt.subplot(3, 3, 8)
metrics = ['Avg Support', 'Avg Confidence', 'Avg Lift', 'Rules Found']
values = [rules_filtered['support'].mean(), 
          rules_filtered['confidence'].mean(),
          rules_filtered['lift'].mean() / 3,  # Normalize for visualization
          len(rules_filtered) / 100]  # Normalize for visualization

plt.bar(metrics, values, color=['blue', 'green', 'orange', 'purple'], alpha=0.7)
plt.ylabel('Score (Normalized)')
plt.title('Cross-selling Performance Metrics')
plt.xticks(rotation=45)

# Plot 9: Item co-occurrence heatmap
plt.subplot(3, 3, 9)
# Select top items for heatmap
top_items = sorted_items[:8]
item_matrix = []
for item1, _ in top_items:
    row = []
    for item2, _ in top_items:
        # Co-occurrence frequency
        co_occurrence = (df[item1] & df[item2]).sum() / len(df)
        row.append(co_occurrence)
    item_matrix.append(row)

item_labels = [item.replace('_', ' ').title()[:8] for item, _ in top_items]
sns.heatmap(item_matrix, xticklabels=item_labels, yticklabels=item_labels, 
            annot=True, fmt='.2f', cmap='Reds', cbar_kws={'label': 'Co-purchase Rate'})
plt.title('Product Co-purchase Heatmap')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('ecommerce_fpgrowth_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nE-commerce cross-selling analysis complete! Plot saved as 'ecommerce_fpgrowth_analysis.png'")