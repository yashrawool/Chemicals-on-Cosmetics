import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pickle

# Streamlit app
st.title("👩‍💻Cosmetic Product Discontinuation Prediction")



# Display an image from a file
st.image(r'C:\Users\Sakshi Gaikwad\Downloads\Discountinued.jpg', caption=None, use_container_width=True)


# Explanation of the app
st.markdown("""
This app shows the discontinuation of a product and if the product is discontinued it will show the date details of the product.""")

# Step 1: Load the dataset
st.subheader("Step 1: Load the Dataset")
st.markdown("""
We load the required dataset. Missing values are handled to ensure the data is clean.
""")
chem = pd.read_csv(r'C:\Users\Sakshi Gaikwad\OneDrive\Desktop\cosmetics\cscpopendata.csv')

# Step 2: Create target variable (1 for discontinued, 0 for not discontinued)
st.subheader("Step 2: Create target variable")
st.markdown("""
Discontinued date column is taken as the target.Assigns 1 if a product has a non-null DiscontinuedDate (indicating it’s discontinued).
Assigns 0 otherwise (indicating it’s still in production).
""")
chem['Discontinued'] = chem['DiscontinuedDate'].notnull().astype(int)

# Step 3: Select features - Use relevant columns, handling missing data where necessary
st.subheader("Step 3: Select features")
st.markdown("""The 'ProductName', 'BrandName', 'PrimaryCategory', 'ChemicalCount' are taken as the features to used as input variables.
""")
features = chem[['ProductName', 'BrandName', 'PrimaryCategory', 'ChemicalCount']]
target = chem['Discontinued']

# Step 4: Handle missing values by filling them with 'Unknown' or other strategy
st.subheader("Step 4: Handle missing values")
st.markdown("""Replaces missing values in the feature columns with "Unknown" to prevent issues during encoding or prediction.
""")
features = features.fillna('Unknown')

# Step 5 :Initialize label encoders
st.subheader("Step 5: Label Encoding")
st.markdown(""" Encodes categorical variables into numerical format using LabelEncoder.
Each column's encoder is stored in a dictionary (label_encoders) for future use.
""")
label_encoders = {}
for column in ['ProductName', 'BrandName', 'PrimaryCategory']:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column])
    label_encoders[column] = le
#Step 6:Split Train-Test and Train the model
st.subheader("Step 6: Train-Test Split and Train the Model")
st.markdown("""Splits the dataset into training (80%) and testing (20%) subsets.
Trains a RandomForestClassifier with 100 decision trees.
""")
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Step 7:Add dates for discontinued products
st.subheader("Step 7:Store Discontinued Dates")
st.markdown("""Filters data for discontinued products (Discontinued == 1).
Stores product names, categories, and relevant date columns for future reference.
""")
discontinued_dates = chem[chem['Discontinued'] == 1][
    ['ProductName', 'BrandName', 'PrimaryCategory', 'InitialDateReported', 'MostRecentDateReported', 'DiscontinuedDate']
]


# Step 8: User input for product prediction
st.subheader("Step 8: Predict Discontinuation for a Product")
st.subheader("⏮️Data Preview")
st.markdown("""Showing the first 50 rows of the dataset""")
# Display the first 50 rows of the updated dataset without encoded values
if st.button("Show First 50 Rows"):
    # Combine features (before encoding) and target column
    original_features = chem[['ProductName', 'BrandName', 'PrimaryCategory', 'ChemicalCount']]
    original_data = pd.concat([original_features, target], axis=1)
    st.write(original_data.head(50))
st.markdown("""
Enter a 'ProductName', 'BrandName', 'PrimaryCategory', 'ChemicalCount' to get know the product is discountinued or not.
""")
product_name = st.text_input("Product Name")
brand_name = st.text_input("Brand Name")
primary_category = st.text_input("Primary Category")
chemical_count = st.number_input("Chemical Count", min_value=0)

if st.button("Predict"):
    # Create DataFrame for user input
    input_data = pd.DataFrame({
        'ProductName': [product_name],
        'BrandName': [brand_name],
        'PrimaryCategory': [primary_category],
        'ChemicalCount': [chemical_count]
    })

    # Encode user input
    for column, le in label_encoders.items():
        if column in input_data.columns:
            input_data[column] = input_data[column].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Make prediction
    prediction = model.predict(input_data)

    # Check if the product is predicted to be discontinued
    if prediction[0] == 1:
        # Find matching dates for the discontinued product
        result = discontinued_dates[
            (discontinued_dates['ProductName'] == product_name) &
            (discontinued_dates['BrandName'] == brand_name) &
            (discontinued_dates['PrimaryCategory'] == primary_category)
        ]

        if not result.empty:
            st.write(f"The product '{product_name}' is likely to be discontinued.")
            st.write("Discontinuation Details:")
            st.write(result)
        else:
            st.write(f"The product '{product_name}' is likely to be discontinued, but no historical date information is available.")
    else:
        st.write(f"The product '{product_name}' is likely to remain in production.")






