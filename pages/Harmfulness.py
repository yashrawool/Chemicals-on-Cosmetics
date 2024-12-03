# # New main
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import streamlit as st

# # Load the dataset
# df = pd.read_csv(r'C:\Users\Raksha Chavan\Downloads\cscpopendata.csv')

# # Fill missing values
# df = df.fillna({'ChemicalName': 'Unknown', 'CasNumbers': 'Unknown', 'ChemicalCount': 0})

# # Rescale the 'ChemicalCount' column for clustering
# scaler = StandardScaler()
# df['ChemicalCountScaled'] = scaler.fit_transform(df[['ChemicalCount']])

# # Fit the KMeans clustering model with the desired number of clusters (e.g., 9 clusters)
# kmeans = KMeans(n_clusters=9, random_state=42)
# df['Cluster'] = kmeans.fit_predict(df[['ChemicalCountScaled']])

# # Group by cluster and calculate the average 'ChemicalCount' for each cluster
# cluster_avg = df.groupby('Cluster')['ChemicalCount'].mean()

# # Sort the clusters by the average 'ChemicalCount'
# sorted_clusters = cluster_avg.sort_values().index

# # Map clusters to harmfulness levels (1 = least harmful, 9 = most harmful)
# harmfulness_map = {cluster: idx + 1 for idx, cluster in enumerate(sorted_clusters)}

# # Map the clusters to harmfulness levels
# df['Harmfulness'] = df['Cluster'].map(harmfulness_map)

# # Create a list of chemicals for each product
# product_chemicals = df.groupby('ProductName').agg(
#     Chemicals=('ChemicalName', lambda x: list(x))
# ).reset_index()

# # Merge the chemicals back into the original dataframe
# df = pd.merge(df[['ProductName', 'ChemicalCount', 'Harmfulness', 'Cluster']], product_chemicals, on='ProductName', how='left')

# # Drop the Cluster column as it's no longer needed
# df = df.drop(columns=['Cluster'])

# # Streamlit Interface
# st.title("Chemical Harmfulness Prediction Based on Chemical Count")

# # User input for product name
# product_name = st.text_input("Enter the Product Name")

# # Predict harmfulness for the entered product
# if st.button("Predict Harmfulness Level"):
#     if product_name:
#         # Find the product in the dataset
#         product_info = df[df['ProductName'] == product_name]
        
#         if not product_info.empty:
#             harmfulness_level = product_info['Harmfulness'].values[0]
#             chemicals_in_product = product_info['Chemicals'].values[0]
            
#             st.subheader(f"Harmfulness Level: {harmfulness_level}")
#             st.write(f"Chemicals in the product: {', '.join(chemicals_in_product)}")
#         else:
#             st.error("Product not found in the dataset. Please check the name and try again.")
#     else:
#         st.error("Please enter a product name.")

# # Show some sample data (optional)
# st.subheader("Sample Data Preview")
# st.write(df.head(50))


# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import streamlit as st

# # Load the dataset
# df = pd.read_csv(r'C:\Users\Raksha Chavan\Downloads\cscpopendata.csv')

# # Fill missing values
# df = df.fillna({'ChemicalName': 'Unknown', 'CasNumbers': 'Unknown', 'ChemicalCount': 0})

# # Rescale the 'ChemicalCount' column for clustering
# scaler = StandardScaler()
# df['ChemicalCountScaled'] = scaler.fit_transform(df[['ChemicalCount']])

# # Fit the KMeans clustering model with the desired number of clusters (e.g., 9 clusters)
# kmeans = KMeans(n_clusters=9, random_state=42)
# df['Cluster'] = kmeans.fit_predict(df[['ChemicalCountScaled']])

# # Group by cluster and calculate the average 'ChemicalCount' for each cluster
# cluster_avg = df.groupby('Cluster')['ChemicalCount'].mean()

# # Sort the clusters by the average 'ChemicalCount'
# sorted_clusters = cluster_avg.sort_values().index

# # Map clusters to harmfulness levels (1 = least harmful, 9 = most harmful)
# harmfulness_map = {cluster: idx + 1 for idx, cluster in enumerate(sorted_clusters)}

# # Map the clusters to harmfulness levels
# df['Harmfulness(Clusters)'] = df['Cluster'].map(harmfulness_map)

# # Create a list of chemicals for each product
# product_chemicals = df.groupby('ProductName').agg(
#     Chemicals=('ChemicalName', lambda x: list(set(x))) #lambda x: list(x))
# ).reset_index()

# # Merge the chemicals back into the original dataframe
# df = pd.merge(df[['ProductName', 'ChemicalCount', 'Harmfulness(Clusters)', 'Cluster']], product_chemicals, on='ProductName', how='left')

# # Drop the Cluster column as it's no longer needed
# df = df.drop(columns=['Cluster'])

# # Streamlit Interface
# st.title("Chemical Harmfulness Prediction Based on Chemical Count")

# # User input for product name
# product_name = st.text_input("Enter the Product Name")

# # Predict harmfulness for the entered product
# if st.button("Predict Harmfulness Level"):
#     if product_name:
#         # Find the product in the dataset
#         product_info = df[df['ProductName'] == product_name]
        
#         if not product_info.empty:
#             harmfulness_level = product_info['Harmfulness(Clusters)'].values[0]
#             chemicals_in_product = product_info['Chemicals'].values[0]
            
#             st.subheader(f"Harmfulness Level: {harmfulness_level}")
#             st.write(f"Chemicals in the product: {', '.join(chemicals_in_product)}")
#         else:
#             st.error("Product not found in the dataset. Please check the name and try again.")
#     else:
#         st.error("Please enter a product name.")

# # Display sample products for each harmfulness level
# st.subheader("Sample Products by Harmfulness Level")

# # User input to set the number of samples per level
# num_samples = st.slider("Number of Sample Products per Harmfulness Level", min_value=1, max_value=5, value=2)

# # Group by harmfulness level and sample
# sample_products = pd.DataFrame()
# for harm_level in sorted(df['Harmfulness(Clusters)'].unique()):
#     harm_group = df[df['Harmfulness(Clusters)'] == harm_level]
#     sample_size = min(num_samples, len(harm_group))
#     sampled_group = harm_group.sample(n=sample_size, random_state=42)
#     sample_products = pd.concat([sample_products, sampled_group])

# if not sample_products.empty:
#     st.dataframe(sample_products[['ProductName', 'ChemicalCount', 'Harmfulness(Clusters)', 'Chemicals']])
# else:
#     st.error("No products available in the dataset.")

# # Optional: Display raw data
# st.subheader("Sample Data Preview")
# st.write(df.head(50))

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Streamlit title
st.title("⚗️Chemical Harmfulness Prediction Based on Chemical Count")

# Explanation of the app
st.markdown("""
This app clusters chemicals based on their count using K-Means clustering. It predicts the harmfulness level of a product based on the average harmfulness of chemicals it contains.
""")

# Step 1: Load the dataset
st.subheader("Step 1: Load the Dataset")
st.markdown("""
We load the required dataset. Missing values are handled to ensure the data is clean.
""")
df = pd.read_csv(r'C:\Users\Sakshi Gaikwad\OneDrive\Desktop\cosmetics\cscpopendata.csv')

# Step 2: Fill missing values
st.subheader("Step 2: Handle Missing Values")
st.markdown("""
Missing values in 'ChemicalName' and 'CasNumbers' columns are replaced with 'Unknown', and zeros are used for missing 'ChemicalCount'.
""")
df = df.fillna({'ChemicalName': 'Unknown', 'CasNumbers': 'Unknown', 'BrandName': 'Unknown' , 'ChemicalCount': 0})

# Step 3: Scale the data
st.subheader("Step 3: Scale the 'ChemicalCount' Column")
st.markdown("""
We scale the 'ChemicalCount' column using StandardScaler to normalize the data, making it suitable for clustering.
""")
scaler = StandardScaler()
df['ChemicalCountScaled'] = scaler.fit_transform(df[['ChemicalCount']])

# Step 4: Perform K-Means clustering
st.subheader("Step 4: Perform K-Means Clustering")
st.markdown("""
Using K-Means, we group the data into clusters based on the scaled 'ChemicalCount'. Each cluster represents a group of similar chemical counts.
""")
kmeans = KMeans(n_clusters=9, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['ChemicalCountScaled']])

# Step 5: Map clusters to harmfulness levels
st.subheader("Step 5: Map Clusters to Harmfulness Levels")
st.markdown("""
Clusters are sorted by the average 'ChemicalCount' in each cluster. We assign a harmfulness level to each cluster (1 = least harmful, 9 = most harmful).
""")
cluster_avg = df.groupby('Cluster')['ChemicalCount'].mean()
sorted_clusters = cluster_avg.sort_values().index
harmfulness_map = {cluster: idx + 1 for idx, cluster in enumerate(sorted_clusters)}
df['Harmfulness(Clusters)'] = df['Cluster'].map(harmfulness_map)

# Step 6: Group chemicals by product
st.subheader("Step 6: Group Chemicals by Product")
st.markdown("""
For each product, we group and list the chemicals it contains.
""")
product_chemicals = df.groupby('ProductName').agg(
    Chemicals=('ChemicalName', lambda x: list(set(x))) #lambda x: list(x))
).reset_index()

# Merge and finalize the dataframe
df = pd.merge(df[['ProductName', 'ChemicalCount', 'Harmfulness(Clusters)', 'Cluster']], product_chemicals, on='ProductName', how='left')
df = df.drop(columns=['Cluster'])

# Step 7: User input for product prediction
st.subheader("Step 7: Predict Harmfulness for a Product")
st.markdown("""
Enter a product name to see its harmfulness level and the chemicals it contains.
""")
product_name = st.text_input("Enter the Product Name")

if st.button("Predict Harmfulness Level"):
    if product_name:
        product_info = df[df['ProductName'] == product_name]
        if not product_info.empty:
            harmfulness_level = product_info['Harmfulness(Clusters)'].values[0]
            chemicals_in_product = product_info['Chemicals'].values[0]
            st.subheader(f"Harmfulness Level: {harmfulness_level}")
            st.write(f"Chemicals in the product: {', '.join(chemicals_in_product)}")
        else:
            st.error("Product not found in the dataset. Please check the name and try again.")
    else:
        st.error("Please enter a product name.")

# Step 8: Display sample products for each harmfulness level
st.subheader("Step 8: Sample Products by Harmfulness Level")
num_samples = st.slider("Number of Sample Products per Harmfulness Level", min_value=1, max_value=5, value=2)
sample_products = pd.DataFrame()

for harm_level in sorted(df['Harmfulness(Clusters)'].unique()):
    harm_group = df[df['Harmfulness(Clusters)'] == harm_level]
    sample_size = min(num_samples, len(harm_group))
    sampled_group = harm_group.sample(n=sample_size, random_state=42)
    sample_products = pd.concat([sample_products, sampled_group])

if not sample_products.empty:
    st.dataframe(sample_products[['ProductName', 'ChemicalCount', 'Harmfulness(Clusters)', 'Chemicals']])
else:
    st.error("No products available in the dataset.")

# Optional: Display raw data
st.subheader("Step 9: Raw Data Preview")
st.markdown("""
View the first 50 rows of the dataset.
""")
st.write(df.head(50))
