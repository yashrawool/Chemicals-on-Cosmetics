import streamlit as st
# import PIL

# img = PIL.

st.set_page_config(
    page_title="Hello",
    page_icon='💄'
)


# Header section
st.write("# Welcome to the Cosmetics Analysis Platform! 💄")
st.write("## Explore Cosmetic Product Predictions 🚀")

st.image(r"C:\Users\Sakshi Gaikwad\Downloads\cosmetic-testing.jpg", caption="Empowering Safer Cosmetic Choices", use_container_width=True)


# Introduction to the project
st.markdown("""
Welcome to our platform, where we provide insights into the cosmetics industry using machine learning models. 
Our project is focused on analyzing cosmetic products to determine two key aspects:

1. **Product Discontinuation Prediction**: Predict whether a cosmetic product is likely to be discontinued based on its features and historical data.
2. **Chemical Harmfulness Prediction**: Assess the harmfulness level of cosmetic products by analyzing the chemicals they contain.
""")

# # Section: Discontinuation Prediction Model
# st.header("🔮 Product Discontinuation Prediction")
# st.markdown("""
# The **Product Discontinuation Prediction Model** uses a **Random Forest Classifier** to analyze historical data about cosmetic products.
# ### Key Features:
# - **Input Features**: Brand name, primary category, chemical count, and product name.
# - **Target**: A binary outcome (`1` for discontinued, `0` for not discontinued).
# - **Process**:
#   - Data preprocessing, including handling missing values and encoding categorical features.
#   - Splitting the data into training and testing sets for model evaluation.
#   - Providing predictions on whether a product is likely to be discontinued.
# ### Outcome:
# The model not only predicts discontinuation but also retrieves relevant dates and history for products that are discontinued. This helps companies and consumers make informed decisions.
# """)

# # Section: Harmfulness Prediction Model
# #st.header("⚗️ Chemical Harmfulness Prediction")
# #st.markdown("""
# #The **Chemical Harmfulness Prediction Model** uses **K-Means Clustering** to categorize products based on the chemicals they contain.
# ### Key Features:
# #- **Input Features**: Chemical count and other chemical-related attributes.
# #- **Clustering**:
#  # - Groups products into clusters based on chemical counts.
#   #- Maps clusters to harmfulness levels (1 = least harmful, 9 = most harmful).
# #- **Chemicals Grouping**: Aggregates a list of chemicals present in each product for detailed analysis.
# ### Outcome:
# #The model predicts the harmfulness level of a product and provides a detailed list of chemicals for transparency. This empowers consumers to make safer choices while using cosmetic products.
# #""")

# Section: Why This Project Matters
# st.header("💡 Why This Project Matters")
# st.markdown("""
# The cosmetics industry is vast, with thousands of products introduced and discontinued every year. 
# - **For Companies**: Understand trends in product discontinuation to better strategize future launches and modifications.
# - **For Consumers**: Gain insights into the safety of products they use daily, ensuring better health and well-being.
# - **For Researchers**: Facilitate further research into harmful chemicals and safe alternatives in cosmetics.
# """)

# # Footer
# st.markdown("---")
# st.write("### Explore the Discontinuation and Harmfulness Prediction Models")
# st.markdown("""
# Use the tabs on the left to navigate through our models and see them in action. 
# Feel free to try predicting for specific products or view sample data from our analysis.
# """)

# st.subheader("Thank you for exploring our project. Together, we can ensure safer and smarter choices in the world of cosmetics!")
