import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

preprocessor = joblib.load(
    r"C:\Users\BLUEZONE\OneDrive\Desktop\Real Estate Investment Advisor\preprocessor.pkl"
)

clf_model = joblib.load(
    r"C:\Users\BLUEZONE\OneDrive\Desktop\Real Estate Investment Advisor\model_classification_rf.pkl"
)

reg_model = joblib.load(
    r"C:\Users\BLUEZONE\OneDrive\Desktop\Real Estate Investment Advisor\model_regression_rf.pkl"
)

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide",
    page_icon="🏠"
)

st.title("🏠 Real Estate Investment Advisor")
st.markdown("### Make smarter property investment decisions with ML-powered insights.")

page = st.sidebar.selectbox(
    "📌 Navigate",
    ["Investment Prediction", "Price Forecast", "Feature Importance", "Locality Insights"]
)

def get_user_input():
    col1, col2 = st.columns(2)

    with col1:
        city = st.text_input("City", "Mumbai")
        locality = st.text_input("Locality", "Andheri")
        property_type = st.selectbox("Property Type", ["Apartment", "Villa", "House"])
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
        size = st.number_input("Size (SqFt)", min_value=100, max_value=10000, value=900)

    with col2:
        price_lakhs = st.number_input("Price (Lakhs)", min_value=1.0, value=75.0)
        year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2015)
        nearby_schools = st.number_input("Nearby Schools", min_value=0, value=2)
        nearby_hospitals = st.number_input("Nearby Hospitals", min_value=0, value=1)
        amenities = st.text_input("Amenities (comma separated)", "Gym,Pool,Clubhouse")

    df = pd.DataFrame([{
        'City': city,
        'Locality': locality,
        'Property_Type': property_type,
        'BHK': bhk,
        'Size_in_SqFt': size,
        'Price_in_Lakhs': price_lakhs,
        'Year_Built': year_built,
        'Nearby_Schools': nearby_schools,
        'Nearby_Hospitals': nearby_hospitals,
        'Amenities': amenities
    }])

    df['Price_per_SqFt'] = df['Price_in_Lakhs'] * 100000 / df['Size_in_SqFt']
    df['Age_of_Property'] = 2025 - df['Year_Built']
    df['Amenities_count'] = df['Amenities'].apply(
        lambda x: len([i for i in x.split(',') if i.strip() != ""])
    )

    df['Furnished_Status'] = "Unknown"
    df['Facing'] = "Unknown"
    df['Owner_Type'] = "Individual"

    return df

if page == "Investment Prediction":
    st.header("📊 Investment Suitability Prediction")

    df_input = get_user_input()
    
    features = [
        'Size_in_SqFt','BHK','Price_per_SqFt','Age_of_Property',
        'Nearby_Schools','Nearby_Hospitals','Amenities_count',
        'Property_Type','Furnished_Status','Facing','Owner_Type',
        'City','Locality'
    ]

    X = df_input[features]
    X_transformed = preprocessor.transform(X)

    if st.button("Predict"):
        pred = clf_model.predict(X)[0]
        prob = clf_model.predict_proba(X)[0][1]

        colA, colB = st.columns(2)

        with colA:
            st.metric(
                "Investment Decision",
                "GOOD INVESTMENT" if pred == 1 else "NOT GOOD",
                f"{prob * 100:.2f}% confidence"
            )

        with colB:
            st.info("✔ Based on RandomForest Classification Model")

if page == "Price Forecast":
    st.header("💰 5-Year Price Forecast")

    df_input = get_user_input()
    features = [
        'Size_in_SqFt','BHK','Price_per_SqFt','Age_of_Property',
        'Nearby_Schools','Nearby_Hospitals','Amenities_count',
        'Property_Type','Furnished_Status','Facing','Owner_Type',
        'City','Locality'
    ]

    X = df_input[features]

    if st.button("Forecast Price"):
        price_pred = reg_model.predict(X)[0]

        st.metric("Estimated Price After 5 Years", f"{price_pred:.2f} Lakhs")

        plt.figure(figsize=(6,4))
        years = [0, 5]
        prices = [df_input['Price_in_Lakhs'][0], price_pred]
        plt.plot(years, prices, marker='o')
        plt.xlabel("Years")
        plt.ylabel("Price (Lakhs)")
        plt.title("5-Year Price Growth Projection")
        st.pyplot(plt)

if page == "Feature Importance":
    st.header("📈 Feature Importance (RandomForest)")

    try:
        importance = clf_model.feature_importances_
        feature_names = clf_model.feature_names_in_

        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False)

        st.dataframe(fi_df)

        plt.figure(figsize=(8,5))
        plt.barh(fi_df["Feature"], fi_df["Importance"])
        plt.gca().invert_yaxis()
        plt.title("Feature Importance")
        st.pyplot(plt)

    except:
        st.error("Feature importance unavailable for this model.")

if page == "Locality Insights":
    st.header("📍 Locality Insights")
    st.info("Upload your dataset to view locality trends.")

    file = st.file_uploader("Upload housing CSV", type=['csv'])

    if file:
        df = pd.read_csv(file)

        if "Locality" in df.columns and "Price_per_SqFt" in df.columns:
            locality_avg = df.groupby("Locality")["Price_per_SqFt"].mean().sort_values(ascending=False)

            st.subheader("Top Localities by Price Per SqFt")
            st.dataframe(locality_avg.head(10))

            plt.figure(figsize=(8,5))
            locality_avg.head(10).plot(kind='bar')
            plt.title("Top 10 Expensive Localities")
            plt.ylabel("Avg Price per SqFt")
            st.pyplot(plt)
        else:
            st.error("Dataset must include Locality & Price_per_SqFt columns.")
