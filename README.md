# Amazon-Recommendation-System
A recommendation system to provide personalized product recommendations based on category preferences, leveraging collaborative filtering techniques and machine learning algorithms. The project includes an interactive web application built using Dash.

Data
The dataset used in this project is available on Kaggle. You can download it from the following link:
Kaggle Dataset Link: **https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset**

After downloading the dataset, place the merged_amazon_data.csv file in the data directory.

## Installation
Clone the repository:
### sh
Copy code
git clone https://github.com/yourusername/Amazon-Recommendation-System.git
Navigate to the project directory:
### sh
Copy code
cd Amazon-Recommendation-System
Install the required dependencies:
### sh
Copy code
pip install -r requirements.txt
# Usage
- Download the dataset from Kaggle and place it in the data directory.
- To run the Dash application, navigate to the src directory and execute the script:
### sh
Copy code
cd src
python app.py
- Open your web browser and go to the provided local URL to interact with the application.
# Project Structure
- data/: Contains the dataset used for training and evaluation (after downloading from Kaggle).
- src/: Contains the main application script.
- notebooks/: Contains the Jupyter notebook for exploratory data analysis.
- README.md: Provides an overview and instructions for the project.
- requirements.txt: Lists the dependencies required to run the project.
- .gitignore: Specifies files and directories to ignore in the repository.
# Technical Stack
- Programming Languages: Python
- Libraries and Frameworks: Pandas, NumPy, scikit-surprise, Dash
- Visualization Tools: Matplotlib, Seaborn
- Data Handling: CSV files
- Machine Learning Techniques: Collaborative Filtering, Singular Value Decomposition (SVD)
# Author
Vachan
(https://github.com/vachan0196)
