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
        available_mrs = data_service.get_mrs_with_plans(month, year)
        st.info(f"📋 Showing {len(available_mrs)} MRs with existing plans for {month:02d}/{year}")
    else:
        available_mrs = st.session_state.mr_data
        st.info(f"📋 Showing all {len(available_mrs)} active MRs")
    
    # Main Content Area
    if available_mrs:
        # Statistics Dashboard
        create_statistics_dashboard(available_mrs)
        
        # MR Selection
        st.header("👥 Medical Representative Selection")
        
        # Multi-select for MRs
        mr_options = [f"{mr['name']} ({mr['territory']})" for mr in available_mrs]
        mr_names = [mr['name'] for mr in available_mrs]
        
        selected_mr_display = st.multiselect(
            "Select Medical Representatives",
            mr_options,
            help="Choose one or more MRs to generate plans for"
        )
        
        # Get actual MR names from selection
        selected_mr_names = []
        for display_name in selected_mr_display:
            for i, option in enumerate(mr_options):
                if option == display_name:
                    selected_mr_names.append(mr_names[i])
                    break
        
        # Processing Section
        if selected_mr_names:
            st.header("🚀 Plan Generation")
            
            # Configuration Summary
            create_config_summary(action, month, year, tier_mix, enable_clustering, 
                                max_clusters, len(selected_mr_names))
            
            # Generate Plans Button
            if st.button("🚀 Generate Enhanced AI Plans", type="primary", use_container_width=True):
                if not st.session_state.processing:
                    st.session_state.processing = True
                    process_plans(
                        selected_mr_names, month, year, action, tier_mix,
                        enable_clustering, max_clusters, data_service, ai_service,
                        max_cluster_switches, min_same_cluster_visits, route_order,
                        min_gap_days, max_visits_without_revenue, min_revenue_threshold,
                        visit_overage_cap, growth_percentage
                    )
                    st.session_state.processing = False
        else:
            st.warning("⚠️ Please select at least one Medical Representative to continue.")
    
    else:
        st.warning("❌ No medical representatives found for the selected criteria.")

def create_statistics_dashboard(mr_data):
    """Create statistics dashboard"""
    st.header("📊 System Overview")
    
    # Calculate statistics
    territories = {}
    states = {}
    for mr in mr_data:
        territory = mr.get('territory', 'Unknown')
        state = mr.get('state', 'Unknown')
        territories[territory] = territories.get(territory, 0) + 1
        states[state] = states.get(state, 0) + 1
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active MRs", len(mr_data))
    
    with col2:
        st.metric("Territories", len(territories))
    
    with col3:
        st.metric("States", len(states))
    
    with col4:
        st.metric("System Status", "🟢 Ready")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Territory distribution
        if territories:
            df_territories = pd.DataFrame(
                list(territories.items())[:8], 
                columns=['Territory', 'Count']
            )
            fig = px.bar(df_territories, x='Territory', y='Count', 
                        title="Top Territories by MR Count")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # State distribution  
        if states:
            df_states = pd.DataFrame(
                list(states.items()), 
                columns=['State', 'Count']
            )
            fig = px.pie(df_states, values='Count', names='State', 
                        title="MR Distribution by State")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def create_config_summary(action, month, year, tier_mix, enable_clustering, 
                         max_clusters, selected_count):
    """Create configuration summary"""
    st.subheader("📋 Configuration Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Planning Details**
        - Action: {action}
        - Period: {month:02d}/{year}
        - Strategy: {tier_mix.title()}
        - Selected MRs: {selected_count}
        """)
    
    with col2:
        clustering_status = "🗺️ Enabled" if enable_clustering else "📍 Disabled"
        st.info(f"""
        **Clustering Config**
        - Status: {clustering_status}
        - Max Clusters/Area: {max_clusters}
        - Route Optimization: ✅
        - Travel Efficiency: ✅
        """)
    
    with col3:
        st.info(f"""
        **Expected Output**
        - Enhanced Plans: {selected_count}
        - ID Mapping: ✅ Enabled
        - Database Storage: ✅ Ready
        - Thread Tracking: ✅ Active
        """)

def process_plans(selected_mr_names, month, year, action, tier_mix, enable_clustering, 
                 max_clusters, data_service, ai_service, max_cluster_switches, 
                 min_same_cluster_visits, route_order, min_gap_days, 
                 max_visits_without_revenue, min_revenue_threshold, 
                 visit_overage_cap, growth_percentage):
    """Process plans with real-time updates"""
    
    st.header("⚡ Processing Plans")
    
    # Create UI config
    ui_config = {
        'max_cluster_switches': max_cluster_switches,
        'min_same_cluster_visits': min_same_cluster_visits,
        'route_order': route_order,
        'min_gap_days': min_gap_days,
        'max_visits_without_revenue': max_visits_without_revenue,
        'min_revenue_threshold': min_revenue_threshold,
        'visit_overage_cap': visit_overage_cap,
        'growth_pct': growth_percentage
    }
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    clustering_status = "🗺️ WITH CLUSTERING" if enable_clustering else "📍 WITHOUT CLUSTERING"
    st.info(f"Starting Enhanced {action} processing {clustering_status}")
    
    results = {'completed': 0, 'errors': 0, 'total_customers': 0, 'total_visits': 0, 'total_revenue': 0}
    
    # Process each MR
    for i, mr_name in enumerate(selected_mr_names):
        try:
            # Update progress
            progress = (i + 1) / len(selected_mr_names)
            progress_bar.progress(progress)
            status_text.text(f"Processing {mr_name} ({i+1}/{len(selected_mr_names)})")
            
            # Load customers
            with st.spinner(f"Loading customers for {mr_name}..."):
                customers = data_service.get_customer_data(mr_name)
            
            if not customers:
                st.error(f"❌ No customers found for {mr_name}")
                results['errors'] += 1
                continue
            
            # Generate plan
            with st.spinner(f"🤖 Generating enhanced plan for {mr_name}..."):
                plan_result = ai_service.generate_monthly_plan_with_clustering(
                    mr_name, month, year, customers, action, tier_mix,
                    data_service, enable_clustering, ui_config
                )
            
            # Save to database
            with st.spinner(f"💾 Saving plan for {mr_name}..."):
                thread_id = plan_result.get('thread_id')
                clustering_metadata = plan_result.get('clustering_metadata', {})
                clean_plan_result = {k: v for k, v in plan_result.items()
                                   if k not in ['thread_id', 'clustering_metadata']}
                
                plan_data = {
                    'mr_name': mr_name,
                    'plan_month': month,
                    'plan_year': year,
                    'original_plan_json': clean_plan_result,
                    'current_plan_json': clean_plan_result,
                    'current_revision': 0 if action == 'NEW_PLAN' else 1,
                    'status': 'ACTIVE',
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'total_customers': len(customers),
                    'total_planned_visits': clean_plan_result['executive_summary']['planned_total_visits'],
                    'total_revenue_target': clean_plan_result['executive_summary']['expected_revenue'],
                    'generation_method': f'ai_enhanced_clustering_v3_{tier_mix}',
                    'data_quality_score': 0.99,
                    'thread_id': thread_id
                }
                
                data_service.save_monthly_plan(plan_data)
            
            # Update results
            visits = clean_plan_result['executive_summary']['planned_total_visits']
            revenue = clean_plan_result['executive_summary']['expected_revenue']
            
            results['completed'] += 1
            results['total_customers'] += len(customers)
            results['total_visits'] += visits
            results['total_revenue'] += revenue
            
            # Show success
            clusters_info = ""
            if clustering_metadata.get('clustering_enabled'):
                total_clusters = clustering_metadata.get('total_clusters', 0)
                total_areas = clustering_metadata.get('total_areas', 0)
                clusters_info = f"🗺️ {total_clusters}C/{total_areas}A"
            
            st.success(f"✅ {mr_name}: {visits} visits, ₹{revenue:,.0f} revenue {clusters_info}")
            
        except Exception as e:
            results['errors'] += 1
            st.error(f"❌ {mr_name}: Processing failed - {str(e)}")
        
        time.sleep(0.5)  # Brief pause for UI updates
    
    # Final summary
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    display_final_summary(results, enable_clustering)

def display_final_summary(results, clustering_enabled):
    """Display final processing summary"""
    st.header("🎉 Processing Complete!")
    
    # Success metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_processed = results['completed'] + results['errors']
    success_rate = f"{(results['completed'] / total_processed * 100):.1f}%" if total_processed > 0 else "0%"
    
    with col1:
        st.metric("Success Rate", success_rate)
    
    with col2:
        st.metric("Total Customers", f"{results['total_customers']:,}")
    
    with col3:
        st.metric("Total Visits", f"{results['total_visits']:,}")
    
    with col4:
        st.metric("Total Revenue", f"₹{results['total_revenue']:,.0f}")
    
    # Status summary
    clustering_badge = "🗺️ Clustering Enabled" if clustering_enabled else "📍 Standard Mode"
    
    st.success(f"""
    **{clustering_badge}**
    
    ✅ {results['completed']} plans completed successfully
    ❌ {results['errors']} errors encountered
    
    All plans saved to database with proper customer code mapping!
    """)
    
    if clustering_enabled:
        st.info("🗺️ **Enhanced with geographical clustering** - Optimized routes and travel efficiency included in all plans")

# Run the app
if __name__ == "__main__":
    main()
