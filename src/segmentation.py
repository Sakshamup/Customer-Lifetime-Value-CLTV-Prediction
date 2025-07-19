from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

def segment_customers(customer_summary, model):
    customer_summary['Predicted_CLTV'] = model.predict(customer_summary[['Recency', 'Frequency', 'First_Purchase']])
    scaler = StandardScaler()
    scaled_cltv = scaler.fit_transform(customer_summary[['Predicted_CLTV']])

    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_summary['Segment'] = kmeans.fit_predict(scaled_cltv)
    return customer_summary
