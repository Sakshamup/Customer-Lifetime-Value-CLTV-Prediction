import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="CLTV Analytics Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    .segment-card {
        background: linear-gradient(135deg, #f59a38 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stSelectbox > div > div {
        background-color: #41d8f2;
        border-radius: 10px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    try:
        # For demo purposes, create sample data if files don't exist
        try:
            model = joblib.load('models/cltv_model.pkl')
            customer_summary = pd.read_csv('customer_cltv_segments.csv')
        except:
            # Create mock data for demonstration
            st.warning("📝 Demo mode: Using sample data. Please upload your model and data files.")
            model = None
            
            # Generate sample customer data
            np.random.seed(42)
            n_customers = 1000
            segments = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk', 'Need Attention']
            
            customer_summary = pd.DataFrame({
                'Customer_ID': range(1, n_customers + 1),
                'Recency': np.random.randint(1, 365, n_customers),
                'Frequency': np.random.randint(1, 50, n_customers),
                'Monetary': np.random.lognormal(4, 1, n_customers),
                'First_Purchase': np.random.randint(30, 1000, n_customers),
                'Segment': np.random.choice(segments, n_customers),
                'Predicted_CLTV': np.random.lognormal(6, 0.8, n_customers)
            })
            
        return model, customer_summary
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

model, customer_summary = load_data()

# Title with emoji and styling
st.markdown('<h1 class="main-header">💰 Customer Lifetime Value Analytics Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.markdown("### 🎯 Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    ["🔮 CLTV Prediction", "📊 Customer Analytics", "🎯 Segment Analysis", "📈 Business Insights"]
)

if customer_summary is not None:
    # Key metrics in sidebar
    st.sidebar.markdown("### 📊 Key Metrics")
    
    total_customers = len(customer_summary)
    avg_cltv = customer_summary['Predicted_CLTV'].mean()
    top_segment = customer_summary['Segment'].value_counts().index[0]
    high_value_customers = (customer_summary['Predicted_CLTV'] > customer_summary['Predicted_CLTV'].quantile(0.8)).sum()
    
    st.sidebar.metric("Total Customers", f"{total_customers:,}")
    st.sidebar.metric("Average CLTV", f"${avg_cltv:,.2f}")
    st.sidebar.metric("Largest Segment", top_segment)
    st.sidebar.metric("High-Value Customers", f"{high_value_customers:,}")

# Main content based on selected page
if page == "🔮 CLTV Prediction":
    st.markdown("## 🔮 Predict Customer Lifetime Value")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Customer Information")
        
        # Input fields with better styling
        recency = st.number_input(
            "📅 Recency (days since last purchase)", 
            min_value=0, 
            max_value=365, 
            value=30,
            help="Number of days since the customer's last purchase"
        )
        
        frequency = st.number_input(
            "🛒 Frequency (number of purchases)", 
            min_value=1, 
            max_value=100, 
            value=5,
            help="Total number of purchases made by the customer"
        )
        
        first_purchase = st.number_input(
            "⏰ Days since first purchase", 
            min_value=0, 
            max_value=1095, 
            value=180,
            help="Number of days since the customer's first purchase"
        )
        
        # Prediction button
        predict_btn = st.button("🎯 Predict CLTV", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Input Visualization")
        
        # Create a radar chart for input visualization
        categories = ['Recency\n(Inverted)', 'Frequency', 'Customer Age']
        values = [
            max(0, 365 - recency) / 365 * 100,  # Invert recency (lower is better)
            min(frequency / 20 * 100, 100),     # Scale frequency
            min(first_purchase / 365 * 100, 100)  # Scale customer age
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Customer Profile',
            line_color='#667eea'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="Customer Profile Radar",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction result
    if predict_btn:
        if model is not None:
            try:
                input_df = pd.DataFrame({
                    'Recency': [recency], 
                    'Frequency': [frequency], 
                    'First_Purchase': [first_purchase]
                })
                prediction = model.predict(input_df)[0]
                
                # Determine customer segment based on CLTV
                if prediction > customer_summary['Predicted_CLTV'].quantile(0.9):
                    segment = "Premium 👑"
                    color = "success"
                elif prediction > customer_summary['Predicted_CLTV'].quantile(0.7):
                    segment = "High Value 🌟"
                    color = "success"
                elif prediction > customer_summary['Predicted_CLTV'].quantile(0.3):
                    segment = "Medium Value 📈"
                    color = "warning"
                else:
                    segment = "Low Value 📉"
                    color = "error"
                
                # Display prediction with styling
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎯 Prediction Result</h2>
                    <h1>${prediction:,.2f}</h1>
                    <h3>Customer Segment: {segment}</h3>
                    <p>Based on the input parameters, this customer's predicted lifetime value is ${prediction:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    percentile = (customer_summary['Predicted_CLTV'] < prediction).mean() * 100
                    st.metric("Percentile Rank", f"{percentile:.1f}%")
                
                with col4:
                    avg_diff = ((prediction - avg_cltv) / avg_cltv) * 100
                    st.metric("vs Average", f"{avg_diff:+.1f}%")
                
                with col5:
                    if recency < 30:
                        status = "Active 🟢"
                    elif recency < 90:
                        status = "Moderate 🟡"
                    else:
                        status = "At Risk 🔴"
                    st.metric("Status", status)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        else:
            # Mock prediction for demo
            mock_prediction = np.random.lognormal(6, 0.3)
            st.markdown(f"""
            <div class="prediction-box">
                <h2>🎯 Demo Prediction Result</h2>
                <h1>${mock_prediction:,.2f}</h1>
                <h3>Customer Segment: Demo Mode</h3>
                <p>This is a demo prediction. Upload your trained model for real predictions.</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "📊 Customer Analytics":
    st.markdown("## 📊 Customer Analytics Overview")
    
    if customer_summary is not None:
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>👥 Total Customers</h3>
                <h2>{len(customer_summary):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_cltv = customer_summary['Predicted_CLTV'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Avg CLTV</h3>
                <h2>${avg_cltv:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_value = customer_summary['Predicted_CLTV'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Total Value</h3>
                <h2>${total_value:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            median_cltv = customer_summary['Predicted_CLTV'].median()
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Median CLTV</h3>
                <h2>${median_cltv:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Interactive charts
        col1, col2 = st.columns(2)
        
        with col1:
            # CLTV distribution
            fig = px.histogram(
                customer_summary, 
                x='Predicted_CLTV', 
                nbins=30,
                title="📊 CLTV Distribution",
                labels={'Predicted_CLTV': 'Predicted CLTV ($)', 'count': 'Number of Customers'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # RFM scatter plot
            fig = px.scatter(
                customer_summary, 
                x='Recency', 
                y='Frequency',
                size='Predicted_CLTV',
                color='Segment',
                title="🎯 RFM Analysis (Recency vs Frequency)",
                labels={'Recency': 'Days Since Last Purchase', 'Frequency': 'Purchase Frequency'},
                hover_data=['Predicted_CLTV']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.markdown("### 🔗 Feature Correlation Matrix")
        numeric_cols = ['Recency', 'Frequency', 'Monetary', 'First_Purchase', 'Predicted_CLTV']
        correlation_data = customer_summary[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_data,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Segment Analysis":
    st.markdown("## 🎯 Customer Segment Analysis")
    
    if customer_summary is not None:
        # Segment selection
        segment = st.selectbox(
            "🎯 Select Customer Segment:", 
            options=['All Segments'] + sorted(customer_summary['Segment'].unique()),
            help="Choose a specific segment to analyze"
        )
        
        if segment == 'All Segments':
            filtered_data = customer_summary
            st.markdown("### 📊 All Segments Overview")
        else:
            filtered_data = customer_summary[customer_summary['Segment'] == segment]
            st.markdown(f"### 📊 {segment} Segment Analysis")
        
        # Segment metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("👥 Customers", f"{len(filtered_data):,}")
        with col2:
            st.metric("💰 Avg CLTV", f"${filtered_data['Predicted_CLTV'].mean():,.0f}")
        with col3:
            percentage = (len(filtered_data) / len(customer_summary)) * 100
            st.metric("📊 % of Total", f"{percentage:.1f}%")
        
        # Charts for segment analysis
        col1, col2 = st.columns(2)
        
        with col1:
            if segment == 'All Segments':
                # Segment comparison
                segment_stats = customer_summary.groupby('Segment')['Predicted_CLTV'].agg(['mean', 'count']).reset_index()
                
                fig = px.bar(
                    segment_stats, 
                    x='Segment', 
                    y='mean',
                    title="📊 Average CLTV by Segment",
                    labels={'mean': 'Average CLTV ($)', 'Segment': 'Customer Segment'},
                    color='mean',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # CLTV distribution for selected segment
                fig = px.histogram(
                    filtered_data, 
                    x='Predicted_CLTV', 
                    nbins=20,
                    title=f"📊 CLTV Distribution - {segment}",
                    labels={'Predicted_CLTV': 'Predicted CLTV ($)', 'count': 'Number of Customers'},
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if segment == 'All Segments':
                # Segment pie chart
                segment_counts = customer_summary['Segment'].value_counts()
                
                fig = px.pie(
                    values=segment_counts.values, 
                    names=segment_counts.index,
                    title="👥 Customer Distribution by Segment",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Box plot for RFM metrics
                rfm_data = filtered_data[['Recency', 'Frequency', 'Monetary']].melt(
                    var_name='Metric', value_name='Value'
                )
                
                fig = px.box(
                    rfm_data, 
                    x='Metric', 
                    y='Value',
                    title=f"📊 RFM Metrics Distribution - {segment}",
                    color='Metric'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Business Insights":
    st.markdown("## 📈 Business Insights & Recommendations")
    
    if customer_summary is not None:
        # Top insights
        st.markdown("### 🎯 Key Insights")
        
        # Calculate insights
        high_value_threshold = customer_summary['Predicted_CLTV'].quantile(0.8)
        high_value_customers = customer_summary[customer_summary['Predicted_CLTV'] > high_value_threshold]
        
        at_risk_customers = customer_summary[
            (customer_summary['Recency'] > 90) & 
            (customer_summary['Predicted_CLTV'] > customer_summary['Predicted_CLTV'].median())
        ]
        
        champions = customer_summary[customer_summary['Segment'] == 'Champions'] if 'Champions' in customer_summary['Segment'].values else pd.DataFrame()
        
        # Insight cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="segment-card">
                <h3>🏆 High-Value Customers</h3>
                <h2>{len(high_value_customers)} customers</h2>
                <p>Represent {(len(high_value_customers)/len(customer_summary)*100):.1f}% of customers but generate 
                ${high_value_customers['Predicted_CLTV'].sum():,.0f} in predicted value</p>
                <strong>💡 Strategy:</strong> VIP treatment, exclusive offers, loyalty programs
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="segment-card">
                <h3>⚠️ At-Risk High-Value Customers</h3>
                <h2>{len(at_risk_customers)} customers</h2>
                <p>High-value customers who haven't purchased in 90+ days</p>
                <strong>💡 Strategy:</strong> Immediate win-back campaigns, personalized offers, direct outreach
            </div>
            """, unsafe_allow_html=True)
        
        # Revenue opportunity analysis
        st.markdown("### 💰 Revenue Opportunity Analysis")
        
        # Create revenue opportunity chart
        segment_stats = customer_summary.groupby('Segment').agg({
            'Predicted_CLTV': ['sum', 'mean', 'count'],
            'Recency': 'mean',
            'Frequency': 'mean'
        }).round(2)
        
        segment_stats.columns = ['Total_CLTV', 'Avg_CLTV', 'Count', 'Avg_Recency', 'Avg_Frequency']
        segment_stats = segment_stats.reset_index()
        
        # Create bubble chart
        fig = px.scatter(
            segment_stats,
            x='Avg_Recency',
            y='Avg_CLTV',
            size='Count',
            color='Total_CLTV',
            hover_name='Segment',
            title="📊 Segment Opportunity Matrix (Size = Customer Count, Color = Total Value)",
            labels={
                'Avg_Recency': 'Average Recency (Days)',
                'Avg_CLTV': 'Average CLTV ($)',
                'Total_CLTV': 'Total Segment Value ($)'
            },
            size_max=60
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Action plan
        st.markdown("### 🎯 Recommended Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 🚀 Growth Opportunities
            - **Upsell to Medium Value customers**
            - **Cross-sell to Loyal Customers**  
            - **Referral programs for Champions**
            - **Premium product launches**
            """)
        
        with col2:
            st.markdown("""
            #### 🛡️ Retention Focus
            - **Win-back campaigns for At-Risk**
            - **Loyalty rewards for Champions**
            - **Personalized offers for High-Value**
            - **Regular engagement touchpoints**
            """)
        
        with col3:
            st.markdown("""
            #### 📊 Optimization Areas
            - **Reduce customer acquisition cost**
            - **Improve onboarding for new customers**
            - **Optimize pricing for segments**
            - **Enhance customer experience**
            """)
        
        # Download section
        st.markdown("### 📥 Export Data")
        
        col1, col2 = st.columns(2)
        with col1:
            csv = customer_summary.to_csv(index=False)
            st.download_button(
                label="📊 Download Customer Data (CSV)",
                data=csv,
                file_name=f"customer_cltv_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            insights_text = f"""
Customer Lifetime Value Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Key Metrics:
- Total Customers: {len(customer_summary):,}
- Average CLTV: ${customer_summary['Predicted_CLTV'].mean():,.2f}
- Total Predicted Value: ${customer_summary['Predicted_CLTV'].sum():,.2f}
- High-Value Customers: {len(high_value_customers)} ({len(high_value_customers)/len(customer_summary)*100:.1f}%)
- At-Risk High-Value: {len(at_risk_customers)}

Segment Distribution:
{customer_summary['Segment'].value_counts().to_string()}
            """
            
            st.download_button(
                label="📄 Download Insights Report (TXT)",
                data=insights_text,
                file_name=f"cltv_insights_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>💰 Customer Lifetime Value Analytics Dashboard | Built with Streamlit & ❤️</p>
    <p>📊 Empower your business with data-driven customer insights</p>
</div>
""", unsafe_allow_html=True)
