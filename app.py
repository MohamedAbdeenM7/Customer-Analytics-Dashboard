import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Analytics App", layout="wide")
model = pickle.load(open("model.pkl","rb"))
st.title("Customer Analytics Dashboard")
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Churn Prediction","Customer Segmentation","Data Visualization"]
)

# =============================
# PAGE 1 : CHURN PREDICTION
# =============================

if page == "Churn Prediction":

    st.header("Customer Churn Prediction")

    col1,col2,col3 = st.columns(3)

    age = col1.slider("Age",18,70,30)
    tenure = col2.slider("Tenure",0,10,3)
    gender = col3.selectbox("Gender",["Male","Female"])

    sex = 1 if gender == "Male" else 0

    if st.button("Predict"):

        input_data = np.array([[age,tenure,sex]])

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("Customer Will Churn")
        else:
            st.success("Customer Will Stay")

        st.write("Prediction Probability:", probability.max())

        prob_df = pd.DataFrame(
            probability,
            columns=["Stay","Churn"]
        )

        st.bar_chart(prob_df)

# =============================
# PAGE 2 : CUSTOMER SEGMENTATION
# =============================

elif page == "Customer Segmentation":

    st.header("Customer Segmentation using K-Means")

    df = pd.read_csv("Mall_Customers.csv")

    X = df[['Annual Income (k$)','Spending Score (1-100)']]

    kmeans = KMeans(n_clusters=5, random_state=42)

    df['Cluster'] = kmeans.fit_predict(X)

    st.write("Clustered Data Preview")
    st.dataframe(df.head())

    fig = plt.figure()

    plt.scatter(
        X.iloc[:,0],
        X.iloc[:,1],
        c=df['Cluster']
    )

    plt.scatter(
        kmeans.cluster_centers_[:,0],
        kmeans.cluster_centers_[:,1],
        s=200
    )

    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")

    plt.title("Customer Segmentation")

    st.pyplot(fig)

    st.subheader("Cluster Distribution")

    st.bar_chart(df['Cluster'].value_counts())

# =============================
# PAGE 3 : DATA VISUALIZATION
# =============================

elif page == "Data Visualization":

    st.header("Customer Data Analysis")

    df = pd.read_csv("Mall_Customers.csv")

    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution")

        fig1 = plt.figure()
        plt.hist(df['Age'], bins=20)
        st.pyplot(fig1)

    with col2:
        st.subheader("Spending Score Distribution")

        fig2 = plt.figure()
        plt.hist(df['Spending Score (1-100)'], bins=20)
        st.pyplot(fig2)

    st.subheader("Income vs Spending Score")

    fig3 = plt.figure()

    plt.scatter(
        df['Annual Income (k$)'],
        df['Spending Score (1-100)']
    )

    plt.xlabel("Income")
    plt.ylabel("Spending Score")

    st.pyplot(fig3)