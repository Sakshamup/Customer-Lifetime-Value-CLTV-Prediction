from src.data_processing import load_and_merge_data, feature_engineering
from src.model_training import train_and_save_model
from src.segmentation import segment_customers
from src.visualization import plot_segments
from src.shap_explain import explain_model
from sklearn.model_selection import train_test_split

def main():
    full_data = load_and_merge_data()
    X, y, customer_summary = feature_engineering(full_data)
    model = train_and_save_model(X, y)
    
    customer_summary = segment_customers(customer_summary, model)
    customer_summary.to_csv('customer_cltv_segments.csv', index=False)

    plot_segments(customer_summary)

    # Optional SHAP explainability on test data
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    explain_model(model, X_test)

if __name__ == "__main__":
    main()
