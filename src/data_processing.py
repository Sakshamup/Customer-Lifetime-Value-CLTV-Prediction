import pandas as pd
from datetime import datetime

def load_and_merge_data():
    orders = pd.read_csv(r'data/olist_orders_dataset.csv')
    customers = pd.read_csv(r'data/olist_customers_dataset.csv')
    payments = pd.read_csv(r'data/olist_order_payments_dataset.csv')

    orders_customers = pd.merge(orders, customers, on='customer_id')
    payment_data = payments.groupby('order_id').payment_value.sum().reset_index()
    full_data = pd.merge(orders_customers, payment_data, on='order_id')

    date_columns = ['order_purchase_timestamp', 'order_delivered_customer_date']
    for col in date_columns:
        full_data[col] = pd.to_datetime(full_data[col], errors='coerce')

    full_data.dropna(subset=['order_purchase_timestamp', 'payment_value'], inplace=True)
    return full_data

def feature_engineering(full_data):
    snapshot_date = datetime(2018, 9, 3)
    customer_summary = full_data.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': [lambda x: (snapshot_date - x.max()).days, 'count', 'min'],
        'payment_value': 'sum'
    }).reset_index()

    customer_summary.columns = ['customer_unique_id', 'Recency', 'Frequency', 'First_Purchase', 'Monetary']
    customer_summary['First_Purchase'] = customer_summary['First_Purchase'].apply(lambda x: (snapshot_date - x).days)

    X = customer_summary[['Recency', 'Frequency', 'First_Purchase']]
    y = customer_summary['Monetary']
    return X, y, customer_summary
