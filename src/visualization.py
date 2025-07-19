import matplotlib.pyplot as plt
import seaborn as sns

def plot_segments(customer_summary):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Segment', y='Predicted_CLTV', data=customer_summary)
    plt.title('Customer Segments by Predicted CLTV')
    plt.show()

