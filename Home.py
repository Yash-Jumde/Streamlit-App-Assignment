import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pickle
import ast

with open("model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("cluster_to_target.pkl", "rb") as f:
    cluster_to_target = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# print(cluster_to_target)


def identify_cluster(data_point, scaler, kmeans):
    norm_data = scaler.transform([data_point])
    print(norm_data)
    cluster_label = kmeans.predict(norm_data)[0]
    assigned_target = cluster_to_target[cluster_label]
    cluster_centroids = kmeans.cluster_centers_
    centroid = cluster_centroids[cluster_label]

    distance = np.linalg.norm(norm_data - centroid, axis=1)

    return f"The data point belongs to cluster {cluster_label} with assigned target {assigned_target} because it is closest to the centroid with a distance: {distance}"

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


# Streamlit app
st.title("Task 1")

result_df = pd.read_csv("./train_predictions.csv")
result_df_csv = convert_df_to_csv(result_df)

test_df = pd.read_csv("./test_predictions.csv")
test_df_csv = convert_df_to_csv(test_df)

col1, col2 = st.columns(2)
with col1:
    st.write("Train data Predictions")
    st.dataframe(result_df, height=400)
    download_button = st.download_button(
    label="Download data as CSV",
    data=result_df_csv,
    file_name='result_df.csv',
    mime='text/csv',
    key="train_data_download"
    )
with col2:
    st.write("Test data Predictions")
    st.dataframe(test_df, height=400)
    download_button = st.download_button(
    label="Download data as CSV",
    data=test_df_csv,
    file_name='test_result_df.csv',
    mime='text/csv',
    key="test_data_download"
    )


# input box for data point
data_input = st.text_input(":blue[Enter the data point array:]", value="")
st.write(":blue[Example input]: [-70, -61, -66, -53, -51, -63, -82, -57, -76, -78,  -66,  -66, -61, -59,  -73, -75, -63, -77]")
# Creating input fields for each feature

# Convert data_point to a Pandas Series

if st.button("Identify Cluster"):
    data_point = ast.literal_eval(data_input)
    data_point_series = pd.Series(data_point, index=[f'T{i}' for i in range(1, 19)])
    # print(data_point_series)
    result = identify_cluster(data_point_series, scaler, kmeans)
    st.write(f":blue[{result}]")




# Load the DataFrame
df = pd.read_csv("./predictions.csv")

st.markdown("<h1 class='h1-text'>Task 2: Predictions</h1>", unsafe_allow_html=True)

st.dataframe(df, height=400)

# Center the download button and DataFrame using the new style
csv = convert_df_to_csv(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='predictions.csv',
    mime='text/csv',
)

# Load data
combined_data = pd.read_csv("./combined_data.csv")

st.markdown("<h1 class='h1-text'>Task 3</h1>", unsafe_allow_html=True)

# Complete data frame
st.dataframe(combined_data)

# Download data
combined_csv = convert_df_to_csv(combined_data)
# Center the download button and DataFrame using the new style
st.download_button(
    label="Download data as CSV",
    data=combined_csv,
    file_name='dataframe.csv',
    mime='text/csv',
)
