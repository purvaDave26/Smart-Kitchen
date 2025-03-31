import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from collections import Counter

class KitchenOperationsOptimizer:
    def __init__(self):
        self.order_history = None
        self.cooking_time_logs = None
        self.waste_logs = None
        self.inventory = None
        self.seasonal_ingredients = None
        self.demand_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def load_data(self, order_file, cooking_file, waste_file, inventory_file, seasonal_file):
        """Load all necessary data files for kitchen optimization"""
        self.order_history = pd.read_csv(order_file)
        self.cooking_time_logs = pd.read_csv(cooking_file)
        self.waste_logs = pd.read_csv(waste_file)
        self.inventory = pd.read_csv(inventory_file)
        self.seasonal_ingredients = pd.read_csv(seasonal_file)
        
        # Convert date columns to datetime
        self.order_history['date'] = pd.to_datetime(self.order_history['date'])
        self.cooking_time_logs['timestamp'] = pd.to_datetime(self.cooking_time_logs['timestamp'])
        self.waste_logs['date'] = pd.to_datetime(self.waste_logs['date'])
        
        print("All data successfully loaded")
        
    def analyze_order_patterns(self, forecast_days=7):
        """
        Analyze historical order patterns and predict future demand for menu items
        Returns a dataframe with predicted quantities for each item
        """
        if self.order_history is None:
            raise ValueError("Order history data not loaded. Please call load_data first.")
            
        # Group orders by date and item
        daily_orders = self.order_history.groupby(['date', 'item_name'])['quantity'].sum().reset_index()
        
        # Create features for prediction
        daily_orders['day_of_week'] = daily_orders['date'].dt.dayofweek
        daily_orders['month'] = daily_orders['date'].dt.month
        daily_orders['day'] = daily_orders['date'].dt.day
        
        # Get unique items
        items = daily_orders['item_name'].unique()
        
        # Create a prediction dataframe
        future_dates = [datetime.now() + timedelta(days=i) for i in range(1, forecast_days+1)]
        future_df_rows = []
        
        for date in future_dates:
            for item in items:
                future_df_rows.append({
                    'date': date,
                    'item_name': item,
                    'day_of_week': date.weekday(),
                    'month': date.month,
                    'day': date.day
                })
                
        future_df = pd.DataFrame(future_df_rows)
        prediction_results = []
        
        # Train a model for each menu item
        for item in items:
            item_data = daily_orders[daily_orders['item_name'] == item]
            
            if len(item_data) > 10:  # Only predict if we have enough data
                X = item_data[['day_of_week', 'month', 'day']]
                y = item_data['quantity']
                
                # Train the model
                self.demand_model.fit(X, y)
                
                # Predict for future dates
                future_item_data = future_df[future_df['item_name'] == item]
                X_pred = future_item_data[['day_of_week', 'month', 'day']]
                predictions = self.demand_model.predict(X_pred)
                
                for i, date in enumerate(future_item_data['date']):
                    prediction_results.append({
                        'date': date,
                        'item_name': item,
                        'predicted_quantity': max(0, round(predictions[i]))
                    })
        
        prediction_df = pd.DataFrame(prediction_results)
        
        # Calculate prep quantities (assume each item needs its ingredients prepped)
        prep_recommendations = prediction_df.groupby(['date', 'item_name'])['predicted_quantity'].sum().reset_index()
        prep_recommendations['prep_buffer'] = (prep_recommendations['predicted_quantity'] * 0.2).round()  # 20% buffer
        prep_recommendations['recommended_prep'] = prep_recommendations['predicted_quantity'] + prep_recommendations['prep_buffer']
        
        return prep_recommendations
    
    def monitor_cooking_processes(self):
        """
        Analyze cooking process logs to identify inefficiencies
        Returns a dataframe with cooking items and their efficiency metrics
        """
        if self.cooking_time_logs is None:
            raise ValueError("Cooking time logs not loaded. Please call load_data first.")
            
        # Calculate average cooking time per dish
        avg_cooking_times = self.cooking_time_logs.groupby('dish_name')['cooking_time_minutes'].agg(
            ['mean', 'min', 'max', 'count', 'std']).reset_index()
        
        # Identify potential inefficiencies
        avg_cooking_times['efficiency_score'] = 100 - (
            (avg_cooking_times['mean'] - avg_cooking_times['min']) / avg_cooking_times['min'] * 100
        )
        
        # Find dishes with high variability
        avg_cooking_times['variability'] = avg_cooking_times['std'] / avg_cooking_times['mean'] * 100
        
        # Flag items with potential issues
        avg_cooking_times['needs_attention'] = (
            (avg_cooking_times['efficiency_score'] < 70) | 
            (avg_cooking_times['variability'] > 30)
        )
        
        # Get specific inefficiencies from cooking process steps
        if 'process_step' in self.cooking_time_logs.columns and 'step_time_minutes' in self.cooking_time_logs.columns:
            step_times = self.cooking_time_logs.groupby(['dish_name', 'process_step'])['step_time_minutes'].agg(
                ['mean', 'count']).reset_index()
            
            # Find steps that take disproportionately long
            dish_step_analysis = []
            for dish in step_times['dish_name'].unique():
                dish_steps = step_times[step_times['dish_name'] == dish]
                total_time = dish_steps['mean'].sum()
                
                for _, step in dish_steps.iterrows():
                    proportion = step['mean'] / total_time * 100
                    dish_step_analysis.append({
                        'dish_name': dish,
                        'process_step': step['process_step'],
                        'avg_time': step['mean'],
                        'proportion_of_total': proportion,
                        'bottleneck': proportion > 40  # Flag steps that take >40% of total time
                    })
            
            step_analysis_df = pd.DataFrame(dish_step_analysis)
            return avg_cooking_times, step_analysis_df
        
        return avg_cooking_times, None
    
    def track_food_waste(self):
        """
        Analyze food waste data to identify patterns and root causes
        Returns a dictionary with waste analysis dataframes
        """
        if self.waste_logs is None:
            raise ValueError("Waste logs not loaded. Please call load_data first.")
            
        # Calculate total waste by category
        waste_by_category = self.waste_logs.groupby('category')['quantity_kg'].sum().reset_index()
        waste_by_category = waste_by_category.sort_values('quantity_kg', ascending=False)
        
        # Analyze waste by reason
        waste_by_reason = self.waste_logs.groupby('reason')['quantity_kg'].sum().reset_index()
        waste_by_reason = waste_by_reason.sort_values('quantity_kg', ascending=False)
        
        # Calculate waste over time to see trends
        waste_over_time = self.waste_logs.groupby('date')['quantity_kg'].sum().reset_index()
        
        # Get weekly average
        waste_over_time['week'] = waste_over_time['date'].dt.isocalendar().week
        waste_over_time['year'] = waste_over_time['date'].dt.year
        weekly_waste = waste_over_time.groupby(['year', 'week'])['quantity_kg'].mean().reset_index()
        
        # Identify items with highest waste
        waste_by_item = self.waste_logs.groupby('item_name')['quantity_kg'].sum().reset_index()
        waste_by_item = waste_by_item.sort_values('quantity_kg', ascending=False)
        
        # Calculate waste value if cost is available
        if 'cost_per_unit' in self.waste_logs.columns:
            self.waste_logs['waste_value'] = self.waste_logs['quantity_kg'] * self.waste_logs['cost_per_unit']
            waste_value_by_item = self.waste_logs.groupby('item_name')['waste_value'].sum().reset_index()
            waste_value_by_item = waste_value_by_item.sort_values('waste_value', ascending=False)
        else:
            waste_value_by_item = None
            
        # Find correlations between waste and day of week
        if len(self.waste_logs) > 0:
            self.waste_logs['day_of_week'] = self.waste_logs['date'].dt.day_name()
            waste_by_day = self.waste_logs.groupby('day_of_week')['quantity_kg'].mean().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
        else:
            waste_by_day = None
            
        # Determine root causes
        root_causes = []
        for item in waste_by_item.head(10)['item_name']:  # Focus on top 10 waste items
            item_waste = self.waste_logs[self.waste_logs['item_name'] == item]
            common_reasons = item_waste['reason'].value_counts().reset_index()
            common_reasons.columns = ['reason', 'count']
            
            root_causes.append({
                'item_name': item,
                'total_waste_kg': item_waste['quantity_kg'].sum(),
                'primary_reason': common_reasons.iloc[0]['reason'] if len(common_reasons) > 0 else 'Unknown',
                'reason_percentage': common_reasons.iloc[0]['count'] / len(item_waste) * 100 if len(common_reasons) > 0 else 0
            })
            
        root_causes_df = pd.DataFrame(root_causes)
            
        return {
            'waste_by_category': waste_by_category,
            'waste_by_reason': waste_by_reason,
            'waste_over_time': waste_over_time,
            'weekly_waste': weekly_waste,
            'waste_by_item': waste_by_item,
            'waste_value_by_item': waste_value_by_item,
            'waste_by_day': waste_by_day,
            'root_causes': root_causes_df
        }
        
    def recommend_menu_adjustments(self):
        """
        Recommend menu adjustments based on ingredient availability,
        seasonality, waste patterns, and popular dishes
        Returns a dataframe with menu recommendations
        """
        if any(df is None for df in [self.order_history, self.inventory, self.seasonal_ingredients, self.waste_logs]):
            raise ValueError("Required data not loaded. Please call load_data first.")
            
        current_month = datetime.now().month
        current_season = self._get_season(current_month)
        
        # Get seasonal ingredients
        seasonal = self.seasonal_ingredients[
            self.seasonal_ingredients['season'] == current_season
        ]['ingredient_name'].tolist()
        
        # Get popular menu items
        recent_orders = self.order_history[
            self.order_history['date'] >= datetime.now() - timedelta(days=90)
        ]
        popular_items = recent_orders.groupby('item_name')['quantity'].sum().reset_index()
        popular_items = popular_items.sort_values('quantity', ascending=False)
        
        # Get low inventory ingredients
        low_inventory = self.inventory[
            self.inventory['quantity'] < self.inventory['reorder_level']
        ]['ingredient_name'].tolist()
        
        # Get high waste ingredients
        high_waste_items = self.waste_logs.groupby('item_name')['quantity_kg'].sum().reset_index()
        high_waste_items = high_waste_items.sort_values('quantity_kg', ascending=False)
        high_waste_ingredients = high_waste_items.head(10)['item_name'].tolist()
        
        # Generate recommendations
        recommendations = []
        
        # 1. Promote seasonal items
        recommendations.append({
            'recommendation_type': 'Promote Seasonal Items',
            'details': f"Feature dishes with {', '.join(seasonal[:5])} which are in season"
        })
        
        # 2. Adjust for low inventory
        if low_inventory:
            recommendations.append({
                'recommendation_type': 'Inventory Adjustments',
                'details': f"Temporarily modify dishes using {', '.join(low_inventory)} or expedite reordering"
            })
        
        # 3. Reduce high waste items
        if high_waste_ingredients:
            recommendations.append({
                'recommendation_type': 'Waste Reduction',
                'details': f"Modify portion sizes or find alternative uses for {', '.join(high_waste_ingredients[:3])}"
            })
        
        # 4. Create specials based on popular items and seasonal ingredients
        top_items = popular_items.head(5)['item_name'].tolist()
        for item in top_items:
            seasonal_variation = f"Seasonal variation of {item} featuring {', '.join(np.random.choice(seasonal, size=min(3, len(seasonal)), replace=False))}"
            recommendations.append({
                'recommendation_type': 'Seasonal Special',
                'details': seasonal_variation
            })
        
        # 5. Price adjustments based on ingredient costs
        if 'cost_per_unit' in self.inventory.columns:
            high_cost_ingredients = self.inventory.sort_values('cost_per_unit', ascending=False).head(5)['ingredient_name'].tolist()
            recommendations.append({
                'recommendation_type': 'Price Adjustment',
                'details': f"Consider adjusting prices for dishes containing {', '.join(high_cost_ingredients)}"
            })
            
        return pd.DataFrame(recommendations)
    
    def _get_season(self, month):
        """Helper function to determine the current season based on month"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def visualize_insights(self):
        """Generate visualizations of key insights"""
        if any(df is None for df in [self.order_history, self.waste_logs]):
            raise ValueError("Required data not loaded. Please call load_data first.")
        
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Visualize order trends
        daily_orders = self.order_history.groupby('date')['quantity'].sum()
        daily_orders.plot(ax=axs[0, 0], title='Daily Order Volume')
        axs[0, 0].set_xlabel('Date')
        axs[0, 0].set_ylabel('Orders')
        
        # 2. Visualize waste by category
        waste_by_category = self.waste_logs.groupby('category')['quantity_kg'].sum().sort_values(ascending=False)
        waste_by_category.plot(kind='bar', ax=axs[0, 1], title='Waste by Category')
        axs[0, 1].set_xlabel('Category')
        axs[0, 1].set_ylabel('Waste (kg)')
        
        # 3. Visualize top menu items
        top_items = self.order_history.groupby('item_name')['quantity'].sum().sort_values(ascending=False).head(10)
        top_items.plot(kind='barh', ax=axs[1, 0], title='Top 10 Menu Items')
        axs[1, 0].set_xlabel('Orders')
        axs[1, 0].set_ylabel('Item')
        
        # 4. Visualize cooking time efficiency
        if self.cooking_time_logs is not None:
            efficiency = self.monitor_cooking_processes()[0]
            efficiency = efficiency.sort_values('efficiency_score')
            efficiency.head(10)['efficiency_score'].plot(kind='barh', ax=axs[1, 1], 
                                             title='Cooking Efficiency Scores (Lower is Less Efficient)')
            axs[1, 1].set_xlabel('Efficiency Score')
            axs[1, 1].set_ylabel('Dish')
        
        plt.tight_layout()
        plt.savefig('kitchen_insights.png')
        plt.close()
        
        print("Visualizations saved to 'kitchen_insights.png'")

    def generate_report(self):
        """Generate a comprehensive kitchen operations report"""
        report = "# Kitchen Operations Optimization Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        # Demand Predictions
        report += "## Demand Predictions\n\n"
        prep_recommendations = self.analyze_order_patterns()
        report += "Top 5 items for tomorrow:\n\n"
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_preps = prep_recommendations[prep_recommendations['date'].dt.date == tomorrow.date()]
        tomorrow_preps = tomorrow_preps.sort_values('predicted_quantity', ascending=False)
        for _, row in tomorrow_preps.head(5).iterrows():
            report += f"- {row['item_name']}: {row['predicted_quantity']} orders (recommend prepping {row['recommended_prep']})\n"
        
        # Cooking Process Efficiency
        report += "\n## Cooking Process Efficiency\n\n"
        efficiency, step_analysis = self.monitor_cooking_processes()
        inefficient_dishes = efficiency[efficiency['needs_attention']].sort_values('efficiency_score')
        report += "Dishes needing process improvements:\n\n"
        for _, row in inefficient_dishes.head(5).iterrows():
            report += f"- {row['dish_name']}: {row['efficiency_score']:.1f}% efficiency, {row['variability']:.1f}% variability\n"
            
        if step_analysis is not None:
            bottlenecks = step_analysis[step_analysis['bottleneck']]
            if len(bottlenecks) > 0:
                report += "\nProcess bottlenecks identified:\n\n"
                for _, row in bottlenecks.head(5).iterrows():
                    report += f"- {row['dish_name']}: {row['process_step']} takes {row['avg_time']:.1f} minutes ({row['proportion_of_total']:.1f}% of total time)\n"
        
        # Food Waste Analysis
        report += "\n## Food Waste Analysis\n\n"
        waste_analysis = self.track_food_waste()
        
        report += "Top waste categories:\n\n"
        for _, row in waste_analysis['waste_by_category'].head(5).iterrows():
            report += f"- {row['category']}: {row['quantity_kg']:.1f} kg\n"
            
        report += "\nTop waste reasons:\n\n"
        for _, row in waste_analysis['waste_by_reason'].head(5).iterrows():
            report += f"- {row['reason']}: {row['quantity_kg']:.1f} kg\n"
            
        report += "\nRoot causes of waste for top items:\n\n"
        for _, row in waste_analysis['root_causes'].head(5).iterrows():
            report += f"- {row['item_name']}: {row['total_waste_kg']:.1f} kg - {row['reason_percentage']:.1f}% due to {row['primary_reason']}\n"
        
        # Menu Recommendations
        report += "\n## Menu Recommendations\n\n"
        menu_recs = self.recommend_menu_adjustments()
        for _, row in menu_recs.iterrows():
            report += f"- {row['recommendation_type']}: {row['details']}\n"
            
        report += "\n## Next Steps\n\n"
        report += "1. Review inefficient cooking processes and implement standardized procedures\n"
        report += "2. Update prep list based on demand predictions\n"
        report += "3. Implement waste reduction strategies for high-waste items\n"
        report += "4. Consider menu adjustments based on seasonality and inventory\n"
        
        with open("kitchen_operations_report.md", "w") as f:
            f.write(report)
            
        print("Report generated and saved to 'kitchen_operations_report.md'")
        return report

# Example usage
if __name__ == "__main__":
    optimizer = KitchenOperationsOptimizer()
    
    # Create sample data for testing
    def create_sample_data():
        # Create dates for the past 90 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        dates = [start_date + timedelta(days=i) for i in range(91)]
        
        # Sample menu items
        menu_items = ['Pasta Carbonara', 'Caesar Salad', 'Grilled Salmon', 
                     'Beef Burger', 'Margherita Pizza', 'Chicken Curry', 
                     'Vegetable Stir Fry', 'Fish and Chips', 'Mushroom Risotto']
        
        # Generate order history
        orders = []
        for date in dates:
            # More orders on weekends
            order_multiplier = 1.5 if date.weekday() >= 5 else 1
            for item in menu_items:
                # Some randomness in popularity
                popularity = np.random.normal(10, 3) * order_multiplier
                quantity = max(0, int(np.random.normal(popularity, max(popularity / 3, 1), 1)[0]))
                if quantity > 0:
                    orders.append({
                        'date': date,
                        'item_name': item,
                        'quantity': quantity,
                        'revenue': quantity * np.random.uniform(8, 25)
                    })
        
        order_df = pd.DataFrame(orders)
        
        # Generate cooking time logs
        cooking_logs = []
        for item in menu_items:
            # Base cooking time for each item
            base_time = np.random.uniform(5, 30)
            # Generate multiple cooking instances
            for _ in range(50):
                # Add some variability to cooking time
                actual_time = max(1, np.random.normal(base_time, base_time/5))
                cooking_logs.append({
                    'timestamp': np.random.choice(dates),
                    'dish_name': item,
                    'cooking_time_minutes': actual_time,
                    'chef_id': np.random.randint(1, 6),
                    'process_step': np.random.choice(['Prep', 'Cook', 'Plate', 'Wait']),
                    'step_time_minutes': actual_time / np.random.uniform(1, 4)
                })
        
        cooking_df = pd.DataFrame(cooking_logs)
        
        # Generate waste logs
        waste_categories = ['Produce', 'Protein', 'Dairy', 'Grains', 'Prepared Food']
        waste_reasons = ['Expired', 'Overproduction', 'Spoiled', 'Trim Waste', 'Customer Return']
        waste_items = ['Tomatoes', 'Lettuce', 'Chicken', 'Beef', 'Rice', 'Pasta', 'Cheese', 'Milk', 'Bread', 'Fish']
        
        waste_logs = []
        for date in dates:
            # Generate 3-7 waste entries per day
            for _ in range(np.random.randint(3, 8)):
                waste_logs.append({
                    'date': date,
                    'category': np.random.choice(waste_categories),
                    'reason': np.random.choice(waste_reasons),
                    'item_name': np.random.choice(waste_items),
                    'quantity_kg': np.random.uniform(0.1, 5.0),
                    'cost_per_unit': np.random.uniform(2, 15)
                })
        
        waste_df = pd.DataFrame(waste_logs)
        
        # Generate inventory data
        inventory = []
        all_ingredients = waste_items + ['Eggs', 'Flour', 'Sugar', 'Oil', 'Spices', 'Potatoes']
        for ingredient in all_ingredients:
            inventory.append({
                'ingredient_name': ingredient,
                'quantity': np.random.uniform(1, 50),
                'unit': np.random.choice(['kg', 'liter', 'units']),
                'reorder_level': np.random.uniform(5, 15),
                'cost_per_unit': np.random.uniform(1, 20)
            })
        
        inventory_df = pd.DataFrame(inventory)
        
        # Generate seasonal ingredients data
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_ingredients = []
        
        winter_ingredients = ['Potatoes', 'Squash', 'Brussels Sprouts', 'Kale', 'Citrus']
        spring_ingredients = ['Asparagus', 'Peas', 'Artichokes', 'Strawberries', 'Rhubarb']
        summer_ingredients = ['Tomatoes', 'Corn', 'Zucchini', 'Berries', 'Watermelon']
        fall_ingredients = ['Apples', 'Pumpkin', 'Sweet Potatoes', 'Cauliflower', 'Mushrooms']
        
        season_to_ingredients = {
            'Winter': winter_ingredients,
            'Spring': spring_ingredients,
            'Summer': summer_ingredients,
            'Fall': fall_ingredients
        }
        
        for season, ingredients in season_to_ingredients.items():
            for ingredient in ingredients:
                seasonal_ingredients.append({
                    'season': season,
                    'ingredient_name': ingredient,
                    'peak_month': np.random.randint(1, 13)
                })
        
        seasonal_df = pd.DataFrame(seasonal_ingredients)
        
        # Save all dataframes to CSV
        order_df.to_csv('order_history.csv', index=False)
        cooking_df.to_csv('cooking_logs.csv', index=False)
        waste_df.to_csv('waste_logs.csv', index=False)
        inventory_df.to_csv('inventory.csv', index=False)
        seasonal_df.to_csv('seasonal_ingredients.csv', index=False)
        
        return order_df, cooking_df, waste_df, inventory_df, seasonal_df
    
    # Create and load sample data
    order_df, cooking_df, waste_df, inventory_df, seasonal_df = create_sample_data()
    
    # Load the data into the optimizer
    optimizer.order_history = order_df
    optimizer.cooking_time_logs = cooking_df
    optimizer.waste_logs = waste_df
    optimizer.inventory = inventory_df
    optimizer.seasonal_ingredients = seasonal_df
    
    # Run the analysis
    print("Analyzing order patterns...")
    prep_recommendations = optimizer.analyze_order_patterns()
    print(f"Generated prep recommendations for {len(prep_recommendations)} items")
    
    print("\nMonitoring cooking processes...")
    efficiency, step_analysis = optimizer.monitor_cooking_processes()
    print(f"Analyzed {len(efficiency)} dishes for cooking efficiency")
    
    print("\nTracking food waste...")
    waste_analysis = optimizer.track_food_waste()
    print(f"Identified {len(waste_analysis['waste_by_category'])} waste categories")
    
    print("\nGenerating menu recommendations...")
    menu_recs = optimizer.recommend_menu_adjustments()
    print(f"Created {len(menu_recs)} menu recommendations")
    
    print("\nVisualizing insights...")
    optimizer.visualize_insights()
    
    print("\nGenerating comprehensive report...")
    optimizer.generate_report()