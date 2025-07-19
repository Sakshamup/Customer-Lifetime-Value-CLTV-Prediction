import shap
import matplotlib.pyplot as plt

def explain_model(model, X_test):
    explainer = shap.Explainer(model.named_steps['gb'])
    X_scaled = model.named_steps['scaler'].transform(X_test)
    shap_values = explainer(X_scaled)
    shap.summary_plot(shap_values, features=X_test, feature_names=X_test.columns)
