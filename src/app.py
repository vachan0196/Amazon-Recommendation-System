import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load the dataset
file_path = 'C:\\Users\\vacha\\Untitled Folder\\Projects\\Recommendation Projects\\Amazon Recommendation system\\merged_amazon_data.csv'
data = pd.read_csv(file_path)

# Clean and preprocess the data
data_cleaned = data.dropna()
data_cleaned['stars'] = data_cleaned['stars'].astype(float)
data_cleaned['price'] = data_cleaned['price'].astype(float)
data_cleaned['listPrice'] = data_cleaned['listPrice'].astype(float)
data_cleaned['log_price'] = np.log1p(data_cleaned['price'])

# Create a placeholder user ID column
data_cleaned['user_id'] = np.arange(len(data_cleaned))

# Prepare the data for Surprise
reader = Reader(rating_scale=(1, 5))
data_surprise = Dataset.load_from_df(data_cleaned[['user_id', 'asin', 'stars']], reader)

# Train the SVD model on the full dataset
trainset = data_surprise.build_full_trainset()
svd = SVD()
svd.fit(trainset)

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Product Recommendation System"),
    dcc.Input(id='category-input', type='text', placeholder='Enter Category Name'),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='recommendations-output')
])

@app.callback(
    Output('recommendations-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [Input('category-input', 'value')]
)
def update_recommendations(n_clicks, category_name):
    if n_clicks > 0 and category_name:
        # Filter the data based on category name
        category_data = data_cleaned[data_cleaned['category_name'].str.contains(category_name, case=False, na=False)]
        
        if category_data.empty:
            return html.Div("No products found for this category.")
        else:
            # Create a placeholder user ID for the category
            category_data['user_id'] = np.arange(len(category_data))

            # Prepare the data for Surprise
            reader = Reader(rating_scale=(1, 5))
            category_surprise = Dataset.load_from_df(category_data[['user_id', 'asin', 'stars']], reader)
            category_trainset = category_surprise.build_full_trainset()

            # Train the SVD model on the category dataset
            svd_category = SVD()
            svd_category.fit(category_trainset)

            # Predict ratings for all items in the category
            dummy_user_id = len(category_data) + 1
            recommendations = []
            
            for item_id in category_trainset.all_items():
                asin = category_trainset.to_raw_iid(item_id)
                estimated_rating = svd_category.predict(dummy_user_id, item_id).est
                recommendations.append((asin, estimated_rating))
            
            recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]

            return html.Div([
                html.H2('Top 10 Recommendations:'),
                html.Ul([html.Li(f'ASIN: {asin}, Estimated Rating: {rating}') for asin, rating in recommendations])
            ])

if __name__ == '__main__':
    app.run_server(debug=True)
