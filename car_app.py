import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go


# Page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="",
    layout="wide"
)


# Custom CSS
st.markdown("""
<style>
main {padding: 2rem 1rem}
h1 {color: #2A9D8F; padding-bottom: 1rem}
</style>
""", unsafe_allow_html=True)


# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("car_prediction_model.pkl")
        return model
    except FileNotFoundError:
        return None


# Header
st.title(" Car Price Prediction System")
st.markdown("### Get instant valuation for your used car")


# Load model
model = load_model()

if model is None:
    st.error("Model file not found!")
    st.info("Run training file first to generate model")
    st.stop()


# Sidebar
st.sidebar.title("Car Details")

year = st.sidebar.slider("Manufacturing Year", 2000, 2024, 2015)

present_price = st.sidebar.number_input(
    "Current Ex-Showroom Price (Lakhs)", 0.0, 50.0, 5.0
)

kms_driven = st.sidebar.number_input(
    "Kilometers Driven", 0, 500000, 50000
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type", ["Petrol", "Diesel", "CNG"]
)

seller_type = st.sidebar.selectbox(
    "Seller Type", ["Dealer", "Individual"]
)

transmission = st.sidebar.selectbox(
    "Transmission", ["Manual", "Automatic"]
)

owner = st.sidebar.selectbox(
    "Number of Previous Owners", [0, 1, 2, 3]
)


# car age
current_year = 2024
car_age = current_year - year


# button
predict_btn = st.sidebar.button("Get Price Estimate")


# Prediction
if predict_btn:

    fuel_encoded = {"Petrol":0,"Diesel":1,"CNG":2}[fuel_type]

    seller_encoded = {"Dealer":0,"Individual":1}[seller_type]

    transmission_encoded = {"Manual":0,"Automatic":1}[transmission]


    input_data = pd.DataFrame({
        "Year":[year],
        "Present_Price":[present_price],
        "Kms_Driven":[kms_driven],
        "Fuel_Type":[fuel_encoded],
        "Seller_Type":[seller_encoded],
        "Transmission":[transmission_encoded],
        "Owner":[owner]
    })


    predicted_price = model.predict(input_data)[0]


    # depreciation
    depreciation = present_price - predicted_price

    depreciation_percent = (
        (depreciation/present_price)*100 if present_price>0 else 0
    )


    st.header("Prediction Result")

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Estimated Selling Price",
        f"{predicted_price:.2f} Lakhs"
    )

    col2.metric(
        "Current Price",
        f"{present_price:.2f} Lakhs"
    )

    col3.metric(
        "Depreciation",
        f"{depreciation:.2f} Lakhs",
        f"{depreciation_percent:.1f}%"
    )


    st.subheader("Price Range")

    lower = predicted_price*0.9
    upper = predicted_price*1.1

    st.success(f"Expected Range ₹ {lower:.2f} - ₹ {upper:.2f}")


    # Gauge chart
    max_price = present_price*1.3

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_price,

        title={'text':"Estimated Price"},

        number={'prefix':"₹","suffix":"L"},

        gauge={
            'axis':{'range':[0,max_price]},

            'bar':{'color':"#2A9D8F"}
        }
    ))

    st.plotly_chart(fig)


    # factors
    st.subheader("Price Factors")

    if car_age<5:
        st.write("New car → higher value")

    if kms_driven<50000:
        st.write("Low km → good price")

    if transmission=="Automatic":
        st.write("Automatic → premium price")


else:

    st.info("Enter details and click button")


    st.subheader("Model Info")

    c1,c2,c3 = st.columns(3)

    c1.metric("Algorithm","Random Forest")

    c2.metric("Accuracy","85%")

    c3.metric("Dataset","300+ cars")