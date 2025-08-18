#!/usr/bin/env python3
"""
Streamlit Web App - AI Visit Planner
===================================
Professional web interface for the Enhanced AI Visit Planner
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import json
import asyncio
import time
import os

# Import your backend classes (NO Jupyter dependencies)
from backend import (
    Config, DataService, EnhancedAIServiceSync, 
    GeographicalClusteringService
)

# Configure Streamlit page
st.set_page_config(
    page_title="AI Visit Planner",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = False
    st.session_state.mr_data = []
    st.session_state.processing = False

# Initialize app
@st.cache_resource
def initialize_app():
    """Initialize the application"""
    try:
        config = Config()
        data_service = DataService(config)
        ai_service = EnhancedAIServiceSync(config)
        return config, data_service, ai_service, True
    except Exception as e:
        st.error(f"Failed to initialize app: {e}")
        return None, None, None, False

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Enhanced AI Visit Planner</h1>
        <p>Professional Edition with Geographical Clustering</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize app
    config, data_service, ai_service, success = initialize_app()
    
    if not success:
        st.error("❌ Application initialization failed. Please check your environment variables.")
        st.info("""
        Required Environment Variables:
        - SUPABASE_URL
        - SUPABASE_ANON_KEY  
        - OPENAI_API_KEY
        - OPENAI_ASSISTANT_ID
        """)
        return
    
    # Load MR data
    if not st.session_state.app_initialized:
        with st.spinner("Loading medical representatives..."):
            try:
                st.session_state.mr_data = data_service.get_medical_representatives()
                st.session_state.app_initialized = True
                st.success(f"✅ Loaded {len(st.session_state.mr_data)} medical representatives")
            except Exception as e:
                st.error(f"Failed to load MR data: {e}")
                return
    
    # Sidebar Configuration
    st.sidebar.header("🎛️ Configuration")
    
    # Action Type
    action = st.sidebar.selectbox(
        "Action Type",
        ["NEW_PLAN", "REVISION"],
        format_func=lambda x: "🆕 New Plan" if x == "NEW_PLAN" else "🔄 Revision"
    )
    
    # Date Selection
    current_date = datetime.now()
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        month = st.selectbox(
            "Month",
            range(1, 13),
            index=current_date.month - 1,
            format_func=lambda x: f"{x:02d} - {datetime(2024, x, 1).strftime('%B')}"
        )
    
    with col2:
        year = st.selectbox(
            "Year", 
            range(2024, 2028),
            index=current_date.year - 2024
        )
    
    # Strategy
    tier_mix = st.sidebar.selectbox(
        "Strategy",
        ["balanced", "growth", "maintenance"],
        format_func=lambda x: x.title()
    )
    
    # Clustering Options
    st.sidebar.subheader("🗺️ Clustering Options")
    enable_clustering = st.sidebar.checkbox("Enable Geo-Clustering", value=True)
    max_clusters = st.sidebar.slider("Max Clusters per Area", 2, 12, 6)
    
    # Advanced Configuration
    with st.sidebar.expander("⚙️ Advanced Configuration"):
        max_cluster_switches = st.slider("Max Cluster Switches/Day", 1, 6, 3)
        min_same_cluster_visits = st.slider("Min Same Cluster Visits", 1, 5, 2)
        route_order = st.selectbox("Route Order", ["nearest", "manual"], 
                                 format_func=lambda x: "Nearest First" if x == "nearest" else "Manual Order")
        
        min_gap_days = st.slider("Min Gap Days (Same Customer)", 3, 14, 7)
        max_visits_without_revenue = st.slider("Max Visits w/o Revenue", 1, 6, 3)
        min_revenue_threshold = st.slider("Min Revenue Threshold (₹)", 500, 5000, 1000, step=250)
        visit_overage_cap = st.slider("Visit Overage Cap", 1.0, 2.0, 1.2, step=0.1)
        growth_percentage = st.slider("Growth Target %", 0.05, 0.30, 0.15, step=0.05)
    
    # Filter MRs based on action
    if action == "REVISION":
        available_mrs = data_service.get_mrs_with_plans(month, year
