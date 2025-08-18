#!/usr/bin/env python3
"""
AI Field Sales Visit Planner - COMPLETE ENHANCED VERSION WITH CLUSTERING
========================================================================
Enhanced production-ready system with geographical clustering, area grouping,
proper ID mapping, and optimal route planning for visit efficiency.

Author: AI Assistant
Version: 3.1.0 - Complete Enhanced Edition
"""

import os
import time
import json
import calendar
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import threading
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Install required packages
try:
    from supabase import create_client, Client
    import openai
    from openai import OpenAI
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import geopy.distance
    import nest_asyncio
except ImportError:
    print("📦 Installing required packages...")
    os.system("pip install supabase openai pandas ipywidgets nest-asyncio scikit-learn geopy")
    from supabase import create_client, Client
    import openai
    from openai import OpenAI
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import geopy.distance
    import nest_asyncio

# Apply nest_asyncio for Jupyter environments
nest_asyncio.apply()

# ================================================================
# CONFIGURATION CLASS
# ================================================================

class Config:
    """Configuration class for database and API connections"""
    
    def __init__(self):
        # Supabase configuration
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        
        # OpenAI configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.assistant_id = os.getenv('OPENAI_ASSISTANT_ID')
        
        if not all([self.supabase_url, self.supabase_key, self.openai_api_key, self.assistant_id]):
            raise Exception("Missing required environment variables. Please set SUPABASE_URL, SUPABASE_ANON_KEY, OPENAI_API_KEY, and OPENAI_ASSISTANT_ID")
        
        # Initialize clients
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        self.openai_client = OpenAI(api_key=self.openai_api_key)

# ================================================================
# DATA SERVICE LAYER
# ================================================================

def fix_plan_data_types(plan_data: Dict[str, Any]) -> Dict[str, Any]:
    """Fix data types for database insertion while preserving thread_id"""
    fixed_data = plan_data.copy()

    integer_fields = [
        'total_customers',
        'total_planned_visits',
        'current_revision'
    ]

    for field in integer_fields:
        if field in fixed_data and fixed_data[field] is not None:
            try:
                fixed_data[field] = int(float(fixed_data[field]))
            except (ValueError, TypeError):
                fixed_data[field] = 0

    if 'total_revenue_target' in fixed_data and fixed_data['total_revenue_target'] is not None:
        try:
            fixed_data['total_revenue_target'] = int(float(fixed_data['total_revenue_target']))
        except (ValueError, TypeError):
            fixed_data['total_revenue_target'] = 0

    if 'data_quality_score' in fixed_data:
        try:
            score = float(fixed_data['data_quality_score'])
            if score > 1.0:
                score = score / 100.0
            fixed_data['data_quality_score'] = round(score, 3)
        except (ValueError, TypeError):
            fixed_data['data_quality_score'] = 0.98

    # Preserve thread_id as string
    if 'thread_id' in plan_data:
        thread_id = plan_data['thread_id']
        if thread_id and isinstance(thread_id, str):
            fixed_data['thread_id'] = thread_id
        else:
            logger.warning(f"Invalid thread_id: {thread_id}")

    return fixed_data

class DataService:
    """Enhanced DataService with holiday support and smart filtering"""

    def __init__(self, config: Config):
        self.supabase = config.supabase
        self.cache = {}

    def get_medical_representatives(self) -> List[Dict[str, Any]]:
        """Get all active medical representatives"""
        try:
            response = self.supabase.table('medical_representatives')\
                .select('id, employee_id, name, phone, email, territory, manager_name, monthly_target, is_active, state')\
                .eq('is_active', True)\
                .order('name')\
                .execute()

            if response.data:
                logger.info(f"Loaded {len(response.data)} active MRs")
                return response.data
            else:
                logger.warning("No active medical representatives found")
                return []

        except Exception as e:
            logger.error(f"Error fetching MRs: {e}")
            raise

    def get_mrs_without_plans(self, month: int, year: int) -> List[Dict[str, Any]]:
        """Get MRs who DON'T have existing plans for given month/year"""
        try:
            response = self.supabase.table('monthly_tour_plans')\
                .select('mr_name')\
                .eq('plan_month', month)\
                .eq('plan_year', year)\
                .eq('status', 'ACTIVE')\
                .execute()

            existing_mr_names = set()
            if response.data:
                existing_mr_names = set([plan['mr_name'] for plan in response.data])
                logger.info(f"Found {len(existing_mr_names)} MRs with existing plans for {month:02d}/{year}")

            all_mrs = self.get_medical_representatives()
            available_mrs = [mr for mr in all_mrs if mr['name'] not in existing_mr_names]

            logger.info(f"{len(available_mrs)} MRs available for {month:02d}/{year}")
            return available_mrs

        except Exception as e:
            logger.error(f"Error fetching available MRs: {e}")
            return self.get_medical_representatives()

    def get_mrs_with_plans(self, month: int, year: int) -> List[Dict[str, Any]]:
        """Get MRs who have existing plans for given month/year"""
        try:
            response = self.supabase.table('monthly_tour_plans')\
                .select('mr_name')\
                .eq('plan_month', month)\
                .eq('plan_year', year)\
                .eq('status', 'ACTIVE')\
                .execute()

            if response.data:
                mr_names = list(set([plan['mr_name'] for plan in response.data]))
                return [mr for mr in self.get_medical_representatives() if mr['name'] in mr_names]

            return []

        except Exception as e:
            logger.error(f"Error fetching MRs with plans: {e}")
            return []

    def get_mr_location_info(self, mr_name: str) -> Dict[str, str]:
        """Get location info for MR"""
        try:
            response = self.supabase.table('medical_representatives')\
                .select('state, territory')\
                .eq('name', mr_name)\
                .single()\
                .execute()

            if response.data:
                state = response.data.get('state', '')
                territory = response.data.get('territory', '')

                state_code_mapping = {
                    'Delhi': 'DL', 'Mumbai': 'MH', 'Bangalore': 'KA',
                    'Chennai': 'TN', 'Kolkata': 'WB', 'Hyderabad': 'TS'
                }

                return {
                    'state_code': state_code_mapping.get(state, state[:2].upper()),
                    'district': territory,
                    'state': state,
                    'territory': territory
                }

            return {'state_code': None, 'district': None, 'state': None, 'territory': None}

        except Exception as e:
            logger.error(f"Error getting MR location: {e}")
            return {'state_code': None, 'district': None, 'state': None, 'territory': None}

    def get_customer_data(self, mr_name: str) -> List[Dict[str, Any]]:
        """Get customer data with enhanced error handling"""
        try:
            response = self.supabase.table('customer_tiers')\
                .select('id, customer_code, customer_name, customer_type, territory, mr_name, area_name, city_name, latitude, longitude, revenue_mtd, revenue_last_month, avg_monthly_revenue, tier_level, tier_score, total_sales_90d, total_orders_90d, conversion_rate_90d, frequency_days, days_since_last_visit, dayparts, channel, kind, credit_blocks, availability_alerts, strict, preferred_visit_day, last_visit_date')\
                .eq('mr_name', mr_name)\
                .execute()

            if not response.data:
                return []

            valid_customers = [c for c in response.data if c.get('latitude') and c.get('longitude')]
            return sorted(valid_customers, key=lambda x: x.get('tier_score', 0) or 0, reverse=True)

        except Exception as e:
            logger.error(f"Error fetching customer data: {e}")
            raise

    def get_holidays_for_period(self, start_date: str, end_date: str,
                               state_code: str = None, district: str = None) -> List[Dict[str, Any]]:
        """Get holidays for specified period"""
        try:
            query = self.supabase.table('holidays')\
                .select('date, name, type, strictness')\
                .gte('date', start_date)\
                .lte('date', end_date)

            if state_code:
                query = query.or_(f'state_code.eq.{state_code},state_code.is.null')

            if district:
                query = query.or_(f'district.eq.{district},district.is.null')

            response = query.order('date').execute()
            return response.data or []

        except Exception as e:
            logger.error(f"Error fetching holidays: {e}")
            return []

    def save_monthly_plan(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save plan with proper revision handling and original plan preservation"""
        try:
            logger.info(f"Saving plan for {plan_data['mr_name']}")

            is_revision = plan_data.get('current_revision', 0) > 0
            fixed_plan_data = fix_plan_data_types(plan_data)

            if is_revision:
                logger.info("Processing REVISION - preserving original plan")

                existing_response = self.supabase.table('monthly_tour_plans')\
                    .select('original_plan_json, current_revision')\
                    .eq('mr_name', plan_data['mr_name'])\
                    .eq('plan_month', plan_data['plan_month'])\
                    .eq('plan_year', plan_data['plan_year'])\
                    .eq('status', 'ACTIVE')\
                    .single()\
                    .execute()

                if existing_response.data:
                    original_plan = existing_response.data.get('original_plan_json')
                    if original_plan:
                        fixed_plan_data['original_plan_json'] = original_plan
                        logger.info("Original plan preserved successfully")
                    else:
                        fixed_plan_data['original_plan_json'] = fixed_plan_data['current_plan_json']
                        logger.warning("No original plan found, using current as original")

                    current_revision = existing_response.data.get('current_revision', 0)
                    fixed_plan_data['current_revision'] = current_revision + 1
                    logger.info(f"Revision incremented to {fixed_plan_data['current_revision']}")
                else:
                    logger.warning("No existing plan found for revision, treating as new plan")
                    fixed_plan_data['original_plan_json'] = fixed_plan_data['current_plan_json']
                    fixed_plan_data['current_revision'] = 1

            else:
                logger.info("Processing NEW PLAN")
                fixed_plan_data['original_plan_json'] = fixed_plan_data['current_plan_json']
                fixed_plan_data['current_revision'] = 0

            try:
                response = self.supabase.table('monthly_tour_plans')\
                    .insert(fixed_plan_data)\
                    .execute()

                if response.data:
                    saved_data = response.data[0]
                    logger.info(f"Plan created successfully for {fixed_plan_data['mr_name']}")
                    return saved_data

            except Exception as insert_error:
                if 'duplicate key' in str(insert_error):
                    logger.info("Plan exists, updating existing record")

                    update_data = {k: v for k, v in fixed_plan_data.items()
                                 if k not in ['mr_name', 'plan_month', 'plan_year', 'status', 'created_at']}

                    update_response = self.supabase.table('monthly_tour_plans')\
                        .update(update_data)\
                        .eq('mr_name', fixed_plan_data['mr_name'])\
                        .eq('plan_month', fixed_plan_data['plan_month'])\
                        .eq('plan_year', fixed_plan_data['plan_year'])\
                        .eq('status', 'ACTIVE')\
                        .execute()

                    if update_response.data:
                        saved_data = update_response.data[0]
                        logger.info(f"Plan updated successfully for {fixed_plan_data['mr_name']}")
                        return saved_data
                    else:
                        raise Exception("Failed to save or update plan - no data returned")
                else:
                    raise insert_error

        except Exception as e:
            logger.error(f"Error saving plan for {plan_data.get('mr_name', 'unknown')}: {e}")
            raise

# ================================================================
# ENHANCED CLUSTERING SERVICE
# ================================================================

class GeographicalClusteringService:
    """Advanced geographical clustering for visit optimization"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.cluster_cache = {}

    def create_geographical_clusters(self, customers: List[Dict[str, Any]],
                                   area_based: bool = True,
                                   max_clusters: int = 8) -> List[Dict[str, Any]]:
        """Create geographical clusters with area grouping and lat/long optimization"""
        try:
            logger.info(f"🗺️ Creating geographical clusters for {len(customers)} customers")

            if not customers:
                return []

            valid_customers = [
                c for c in customers
                if c.get('latitude') and c.get('longitude') and
                   float(c['latitude']) != 0 and float(c['longitude']) != 0
            ]

            logger.info(f"📍 {len(valid_customers)} customers have valid coordinates")

            if len(valid_customers) < 2:
                for i, customer in enumerate(customers):
                    customer['cluster_id'] = f"C001"
                    customer['cluster_center_lat'] = customer.get('latitude', 0)
                    customer['cluster_center_lng'] = customer.get('longitude', 0)
                    customer['cluster_size'] = len(customers)
                return customers

            enhanced_customers = customers.copy()

            if area_based:
                enhanced_customers = self._cluster_by_area(enhanced_customers, max_clusters)
            else:
                enhanced_customers = self._cluster_by_coordinates(enhanced_customers, max_clusters)

            enhanced_customers = self._add_cluster_metadata(enhanced_customers)
            self._calculate_cluster_efficiency(enhanced_customers)

            logger.info(f"✅ Clustering completed with optimization metrics")
            return enhanced_customers

        except Exception as e:
            logger.error(f"❌ Clustering failed: {e}")
            return self._fallback_clustering(customers)

    def _cluster_by_area(self, customers: List[Dict], max_clusters: int) -> List[Dict]:
        """Cluster customers within each geographical area"""
        try:
            area_groups = {}
            for customer in customers:
                area = customer.get('area_name', 'Unknown_Area')
                if area not in area_groups:
                    area_groups[area] = []
                area_groups[area].append(customer)

            logger.info(f"📍 Found {len(area_groups)} distinct areas")

            clustered_customers = []
            cluster_counter = 1

            for area_name, area_customers in area_groups.items():
                logger.info(f"🏘️ Clustering area '{area_name}' with {len(area_customers)} customers")

                if len(area_customers) <= 2:
                    for customer in area_customers:
                        customer['cluster_id'] = f"C{cluster_counter:03d}"
                        customer['area_cluster'] = f"{area_name}_C{cluster_counter:03d}"
                    cluster_counter += 1
                else:
                    area_clustered = self._apply_kmeans_to_group(
                        area_customers,
                        max_clusters,
                        cluster_counter,
                        area_name
                    )
                    cluster_counter += len(set([c.get('cluster_id') for c in area_clustered]))

                clustered_customers.extend(area_customers)

            return clustered_customers

        except Exception as e:
            logger.error(f"❌ Area-based clustering failed: {e}")
            return self._cluster_by_coordinates(customers, max_clusters)

    def _cluster_by_coordinates(self, customers: List[Dict], max_clusters: int) -> List[Dict]:
        """Direct coordinate-based clustering"""
        try:
            coordinates = []
            valid_customers = []

            for customer in customers:
                lat = customer.get('latitude', 0)
                lng = customer.get('longitude', 0)

                if lat and lng and float(lat) != 0 and float(lng) != 0:
                    coordinates.append([float(lat), float(lng)])
                    valid_customers.append(customer)

            if len(coordinates) < 2:
                return self._fallback_clustering(customers)

            optimal_clusters = min(max_clusters, max(2, len(coordinates) // 3))

            coordinates_array = np.array(coordinates)
            scaler = StandardScaler()
            scaled_coords = scaler.fit_transform(coordinates_array)

            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_coords)

            for i, customer in enumerate(valid_customers):
                cluster_id = f"C{cluster_labels[i] + 1:03d}"
                customer['cluster_id'] = cluster_id
                customer['area_cluster'] = f"{customer.get('area_name', 'Area')}_{cluster_id}"

            unassigned_customers = [c for c in customers if c not in valid_customers]
            for customer in unassigned_customers:
                customer['cluster_id'] = "C001"
                customer['area_cluster'] = f"{customer.get('area_name', 'Area')}_C001"

            return customers

        except Exception as e:
            logger.error(f"❌ Coordinate clustering failed: {e}")
            return self._fallback_clustering(customers)

    def _apply_kmeans_to_group(self, group_customers: List[Dict], max_clusters: int,
                              cluster_start_id: int, area_name: str) -> List[Dict]:
        """Apply K-Means clustering to a specific group"""
        try:
            coordinates = []
            for customer in group_customers:
                lat = float(customer.get('latitude', 0))
                lng = float(customer.get('longitude', 0))
                if lat != 0 and lng != 0:
                    coordinates.append([lat, lng])

            if len(coordinates) < 2:
                cluster_id = f"C{cluster_start_id:03d}"
                for customer in group_customers:
                    customer['cluster_id'] = cluster_id
                    customer['area_cluster'] = f"{area_name}_{cluster_id}"
                return group_customers

            n_clusters = min(max_clusters, max(2, len(coordinates) // 2))

            coordinates_array = np.array(coordinates)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coordinates_array)

            coord_index = 0
            for customer in group_customers:
                lat = float(customer.get('latitude', 0))
                lng = float(customer.get('longitude', 0))

                if lat != 0 and lng != 0:
                    cluster_id = f"C{cluster_start_id + cluster_labels[coord_index]:03d}"
                    coord_index += 1
                else:
                    cluster_id = f"C{cluster_start_id:03d}"

                customer['cluster_id'] = cluster_id
                customer['area_cluster'] = f"{area_name}_{cluster_id}"

            return group_customers

        except Exception as e:
            logger.error(f"❌ Group clustering failed: {e}")
            cluster_id = f"C{cluster_start_id:03d}"
            for customer in group_customers:
                customer['cluster_id'] = cluster_id
                customer['area_cluster'] = f"{area_name}_{cluster_id}"
            return group_customers

    def _add_cluster_metadata(self, customers: List[Dict]) -> List[Dict]:
        """Add cluster center coordinates and size information"""
        try:
            clusters = {}
            for customer in customers:
                cluster_id = customer.get('cluster_id', 'C001')
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(customer)

            for cluster_id, cluster_customers in clusters.items():
                valid_coords = [
                    (float(c.get('latitude', 0)), float(c.get('longitude', 0)))
                    for c in cluster_customers
                    if c.get('latitude') and c.get('longitude') and
                       float(c.get('latitude', 0)) != 0 and float(c.get('longitude', 0)) != 0
                ]

                if valid_coords:
                    center_lat = sum(coord[0] for coord in valid_coords) / len(valid_coords)
                    center_lng = sum(coord[1] for coord in valid_coords) / len(valid_coords)
                else:
                    center_lat = 28.6139  # Delhi default
                    center_lng = 77.2090

                for customer in cluster_customers:
                    customer['cluster_center_lat'] = round(center_lat, 6)
                    customer['cluster_center_lng'] = round(center_lng, 6)
                    customer['cluster_size'] = len(cluster_customers)

                    if customer.get('latitude') and customer.get('longitude'):
                        try:
                            distance = geopy.distance.geodesic(
                                (center_lat, center_lng),
                                (float(customer['latitude']), float(customer['longitude']))
                            ).kilometers
                            customer['distance_from_center_km'] = round(distance, 2)
                        except:
                            customer['distance_from_center_km'] = 0
                    else:
                        customer['distance_from_center_km'] = 0

            return customers

        except Exception as e:
            logger.error(f"❌ Adding cluster metadata failed: {e}")
            return customers

    def _calculate_cluster_efficiency(self, customers: List[Dict]):
        """Calculate and log cluster efficiency metrics"""
        try:
            clusters = {}
            for customer in customers:
                cluster_id = customer.get('cluster_id', 'C001')
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(customer)

            total_intra_distance = 0
            total_customers = 0

            for cluster_id, cluster_customers in clusters.items():
                if len(cluster_customers) < 2:
                    continue

                distances = []
                for i, c1 in enumerate(cluster_customers):
                    for j, c2 in enumerate(cluster_customers[i+1:], i+1):
                        if (c1.get('latitude') and c1.get('longitude') and
                            c2.get('latitude') and c2.get('longitude')):
                            try:
                                dist = geopy.distance.geodesic(
                                    (float(c1['latitude']), float(c1['longitude'])),
                                    (float(c2['latitude']), float(c2['longitude']))
                                ).kilometers
                                distances.append(dist)
                            except:
                                continue

                if distances:
                    avg_distance = sum(distances) / len(distances)
                    total_intra_distance += avg_distance * len(cluster_customers)
                    total_customers += len(cluster_customers)

                    logger.info(f"📊 Cluster {cluster_id}: {len(cluster_customers)} customers, "
                              f"avg intra-distance: {avg_distance:.2f}km")

            if total_customers > 0:
                overall_efficiency = total_intra_distance / total_customers
                logger.info(f"🎯 Overall cluster efficiency: {overall_efficiency:.2f}km avg distance")

        except Exception as e:
            logger.error(f"❌ Efficiency calculation failed: {e}")

    def _fallback_clustering(self, customers: List[Dict]) -> List[Dict]:
        """Fallback clustering when advanced methods fail"""
        logger.warning("⚠️ Using fallback clustering method")

        for i, customer in enumerate(customers):
            cluster_id = f"C{(i // 5) + 1:03d}"  # 5 customers per cluster
            customer['cluster_id'] = cluster_id
            customer['area_cluster'] = f"{customer.get('area_name', 'Area')}_{cluster_id}"
            customer['cluster_center_lat'] = customer.get('latitude', 28.6139)
            customer['cluster_center_lng'] = customer.get('longitude', 77.2090)
            customer['cluster_size'] = min(5, len(customers) - (i // 5) * 5)
            customer['distance_from_center_km'] = 0

        return customers

# ================================================================
# BASE AI SERVICE CLASS
# ================================================================

class RealAIService:
    """Base AI service that EnhancedAIService inherits from"""
    
    def __init__(self, config: Config):
        self.config = config
        self.openai_client = config.openai_client
        self.assistant_id = config.assistant_id
        self.data_service = None

    def prepare_customer_data(self, customers: List[Dict[str, Any]], 
                            action: str = "NEW_PLAN", 
                            month: int = None, 
                            year: int = None) -> tuple[str, Dict[str, str]]:
        """Base method for preparing customer data (fallback)"""
        try:
            csv_lines = []
            id_mapping = {}
            
            if action == "NEW_PLAN":
                header = "id,last_visit_date,revenue_mtd,revenue_last_month,avg_monthly_revenue,latitude,longitude,tier_code,channel,kind,frequency_days,area_name,dayparts,preferred_visit_day,strict"
                csv_lines.append(header)
                
                for customer in customers:
                    db_id = customer.get('id', customer.get('customer_code', ''))
                    customer_code = customer.get('customer_code', '')
                    id_mapping[str(db_id)] = customer_code
                    
                    line = f"{db_id},{customer.get('last_visit_date', '1900-01-01')},0,0,0,0,0,3,Doctor,,30,Area1,AM,Mon,False"
                    csv_lines.append(line)
            
            return '\n'.join(csv_lines), id_mapping
            
        except Exception as e:
            logger.error(f"Base CSV preparation failed: {e}")
            return "", {}

    async def _call_openai_assistant_with_retry(self, ai_input: Dict, existing_thread_id: str = None) -> tuple[Dict, str]:
        """Call OpenAI assistant with retry logic"""
        try:
            if existing_thread_id:
                thread_id = existing_thread_id
                logger.info(f"Using existing thread: {thread_id}")
            else:
                thread = self.openai_client.beta.threads.create()
                thread_id = thread.id
                logger.info(f"Created new thread: {thread_id}")

            message = self.openai_client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=json.dumps(ai_input)
            )

            run = self.openai_client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id
            )

            while run.status in ['queued', 'in_progress', 'cancelling']:
                await asyncio.sleep(1)
                run = self.openai_client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )

            if run.status == 'completed':
                messages = self.openai_client.beta.threads.messages.list(
                    thread_id=thread_id
                )
                
                response_content = messages.data[0].content[0].text.value
                
                try:
                    plan_result = json.loads(response_content)
                    return plan_result, thread_id
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse AI response as JSON: {e}")
                    raise Exception(f"Invalid JSON response from AI: {response_content[:200]}...")
            else:
                raise Exception(f"AI assistant run failed with status: {run.status}")

        except Exception as e:
            logger.error(f"OpenAI assistant call failed: {e}")
            raise

# ================================================================
# ENHANCED AI SERVICE WITH CLUSTERING
# ================================================================

class EnhancedAIService(RealAIService):
    """Enhanced AI service with clustering support - Standard CSV Format"""

    def __init__(self, config: Config):
        super().__init__(config)
        self.clustering_service = GeographicalClusteringService()

    def prepare_customer_data_with_clustering(self, customers: List[Dict[str, Any]],
                                            action: str = "NEW_PLAN",
                                            month: int = None,
                                            year: int = None,
                                            enable_clustering: bool = True,
                                            max_clusters_per_area: int = 6) -> tuple[str, Dict[str, str], Dict[str, Any]]:
        """Enhanced data preparation with clustering - Standard CSV Format"""
        try:
            logger.info(f"📋 Preparing {action} CSV with clustering for {len(customers)} customers")

            # Apply clustering if enabled
            if enable_clustering:
                logger.info("🗺️ Applying geographical clustering...")
                customers = self.clustering_service.create_geographical_clusters(
                    customers,
                    area_based=True,
                    max_clusters=max_clusters_per_area
                )

            # Prepare standard mappings
            day_mapping = {
                'Monday': 'Mon', 'Tuesday': 'Tue', 'Wednesday': 'Wed',
                'Thursday': 'Thu', 'Friday': 'Fri', 'Saturday': 'Sat', 'Sunday': 'Sun'
            }

            tier_mapping = {
                'TIER_1_CHAMPION': 1, 'TIER_2_PERFORMER': 2,
                'TIER_3_DEVELOPER': 3, 'TIER_4_PROSPECT': 4
            }

            csv_lines = []
            id_mapping = {}  # Maps DB_ID -> CUSTOMER_CODE
            clustering_metadata = {
                'total_clusters': len(set(c.get('cluster_id', 'C001') for c in customers)),
                'total_areas': len(set(c.get('area_name', 'Unknown') for c in customers)),
                'clustering_enabled': enable_clustering
            }

            def safe_value(val, default='', is_revenue=False):
                if val is None or val == '':
                    return str(default)
                if is_revenue:
                    try:
                        return str(round(float(val)))
                    except (ValueError, TypeError):
                        return str(default)
                return str(val)

            if action == "NEW_PLAN":
                header = "id,last_visit_date,revenue_mtd,revenue_last_month,avg_monthly_revenue,latitude,longitude,tier_code,channel,kind,frequency_days,area_name,dayparts,preferred_visit_day,strict,cluster_id"
                csv_lines.append(header)

                for customer in customers:
                    db_id = customer.get('id', customer.get('customer_code', ''))
                    customer_code = customer.get('customer_code', '')
                    
                    if not db_id or not customer_code:
                        logger.warning(f"Missing ID or code: db_id={db_id}, customer_code={customer_code}")
                        continue
                    
                    id_mapping[str(db_id)] = customer_code
                    logger.debug(f"Mapping: DB_ID {db_id} -> CUSTOMER_CODE {customer_code}")

                    tier_code = tier_mapping.get(customer.get('tier_level', 'TIER_3_DEVELOPER'), 3)
                    preferred_day = customer.get('preferred_visit_day', 'Monday')
                    short_day = day_mapping.get(preferred_day, 'Mon')
                    customer_type = customer.get('customer_type', 'Doctor')

                    line = (
                        f"{db_id},"
                        f"{safe_value(customer.get('last_visit_date'), '1900-01-01')},"
                        f"{safe_value(customer.get('revenue_mtd', 0), 0, is_revenue=True)},"
                        f"{safe_value(customer.get('revenue_last_month', 0), 0, is_revenue=True)},"
                        f"{safe_value(customer.get('avg_monthly_revenue', 0), 0, is_revenue=True)},"
                        f"{safe_value(customer.get('latitude', 0), 0)},"
                        f"{safe_value(customer.get('longitude', 0), 0)},"
                        f"{tier_code},"
                        f"{customer_type},"
                        f","
                        f"{safe_value(customer.get('frequency_days', 30), 30)},"
                        f"{safe_value(customer.get('area_name'), 'Area1')},"
                        f"{safe_value(customer.get('dayparts'), 'AM')},"
                        f"{short_day},"
                        f"{safe_value(customer.get('strict', False), False)},"
                        f"{safe_value(customer.get('cluster_id'), 'C001')}"
                    )
                    csv_lines.append(line)

            elif action == "REVISION":
                header = "id,last_visit_date,revenue_mtd,cluster_id,area_name"
                csv_lines.append(header)

                visited_customers = self._filter_visited_customers(customers, month, year)
                logger.info(f"📊 REVISION: Found {len(visited_customers)} visited customers")

                for customer in visited_customers:
                    db_id = customer.get('id', customer.get('customer_code', ''))
                    customer_code = customer.get('customer_code', '')
                    
                    if not db_id or not customer_code:
                        logger.warning(f"Missing ID or code: db_id={db_id}, customer_code={customer_code}")
                        continue
                    
                    id_mapping[str(db_id)] = customer_code

                    line = (
                        f"{db_id},"
                        f"{customer.get('last_visit_date', '1900-01-01')},"
                        f"{safe_value(customer.get('revenue_mtd', 0), 0, is_revenue=True)},"
                        f"{safe_value(customer.get('cluster_id'), 'C001')},"
                        f"{safe_value(customer.get('area_name'), 'Area1')}"
                    )
                    csv_lines.append(line)

            csv_data = '\n'.join(csv_lines)
            logger.info(f"✅ Standard CSV {action} prepared: {len(csv_lines)-1} rows with clustering")
            logger.info(f"🔍 ID Mapping created: {len(id_mapping)} entries")

            return csv_data, id_mapping, clustering_metadata

        except Exception as e:
            logger.error(f"❌ Enhanced CSV preparation failed: {e}")
            csv_data, id_mapping = self.prepare_customer_data(customers, action, month, year)
            return csv_data, id_mapping, {'clustering_enabled': False, 'error': str(e)}

    def _filter_visited_customers(self, customers: List[Dict], month: int, year: int) -> List[Dict]:
        """Filter customers who were visited in the specific month/year"""
        if not month or not year:
            return customers

        target_month_start = date(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        target_month_end = date(year, month, last_day)

        visited_customers = []
        for customer in customers:
            last_visit = customer.get('last_visit_date')
            if last_visit and last_visit != '1900-01-01':
                try:
                    visit_date = datetime.strptime(last_visit, '%Y-%m-%d').date()
                    if target_month_start <= visit_date <= target_month_end:
                        visited_customers.append(customer)
                except (ValueError, TypeError):
                    continue

        return visited_customers

    def _map_ids_back_to_codes(self, plan_result: Dict[str, Any], id_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Map database IDs back to customer codes in the AI plan result"""
        try:
            logger.info(f"🔄 Mapping {len(id_mapping)} IDs back to customer codes")
            
            mapped_plan = json.loads(json.dumps(plan_result))
            mapped_count = 0
            unmapped_count = 0
            
            if 'daily_plan' in mapped_plan:
                for day in mapped_plan['daily_plan']:
                    if 'dayparts' in day:
                        for daypart, customer_ids in day['dayparts'].items():
                            if isinstance(customer_ids, list):
                                mapped_ids = []
                                for customer_id in customer_ids:
                                    customer_id_str = str(customer_id)
                                    if customer_id_str in id_mapping:
                                        customer_code = id_mapping[customer_id_str]
                                        mapped_ids.append(customer_code)
                                        mapped_count += 1
                                        logger.debug(f"Mapped: {customer_id} -> {customer_code}")
                                    else:
                                        mapped_ids.append(customer_id)
                                        unmapped_count += 1
                                        logger.warning(f"No mapping found for ID: {customer_id}")
                                
                                day['dayparts'][daypart] = mapped_ids
            
            if 'revision_notes' in mapped_plan:
                revision_notes = mapped_plan['revision_notes']
                for db_id, customer_code in id_mapping.items():
                    revision_notes = revision_notes.replace(str(db_id), customer_code)
                mapped_plan['revision_notes'] = revision_notes
            
            logger.info(f"✅ ID mapping completed: {mapped_count} mapped, {unmapped_count} unmapped")
            return mapped_plan
            
        except Exception as e:
            logger.error(f"❌ ID mapping failed: {e}")
            return plan_result

    def _get_existing_thread_id(self, data_service: DataService, mr_name: str, month: int, year: int) -> Optional[str]:
        """Get existing thread_id for revision"""
        try:
            response = data_service.supabase.table('monthly_tour_plans')\
                .select('thread_id')\
                .eq('mr_name', mr_name)\
                .eq('plan_month', month)\
                .eq('plan_year', year)\
                .eq('status', 'ACTIVE')\
                .single()\
                .execute()

            if response.data and response.data.get('thread_id'):
                thread_id = response.data['thread_id']
                logger.info(f"📎 Found existing thread_id: {thread_id}")
                return thread_id
            else:
                logger.info("📎 No existing thread_id found for revision")
                return None

        except Exception as e:
            logger.warning(f"⚠️ Could not fetch existing thread_id: {e}")
            return None

    def _validate_ai_response(self, plan_result: Dict[str, Any], customers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate AI response and ensure all customer codes are valid"""
        try:
            valid_customer_codes = set()
            for customer in customers:
                customer_code = customer.get('customer_code', '')
                if customer_code:
                    valid_customer_codes.add(customer_code)

            if 'daily_plan' in plan_result:
                for day in plan_result['daily_plan']:
                    if 'dayparts' in day:
                        for daypart, customer_ids in day['dayparts'].items():
                            if isinstance(customer_ids, list):
                                valid_ids = []
                                for customer_id in customer_ids:
                                    if str(customer_id) in valid_customer_codes:
                                        valid_ids.append(customer_id)
                                    else:
                                        logger.warning(f"⚠️ Invalid customer code in plan: {customer_id}")
                                
                                day['dayparts'][daypart] = valid_ids

            logger.info("✅ AI response validation completed")
            return plan_result

        except Exception as e:
            logger.error(f"❌ AI response validation failed: {e}")
            return plan_result

    def create_enhanced_ai_metadata(self, mr_name: str, month: int, year: int,
                                  customers: List[Dict], action: str, tier_mix: str,
                                  holidays: List[Dict] = None,
                                  clustering_metadata: Dict = None,
                                  ui_config: Dict = None) -> Dict:
        """Create enhanced metadata with UI configuration values"""

        territory = customers[0].get('territory', 'Unknown') if customers else 'Unknown'
        today = date.today()

        if action == "NEW_PLAN" and date(year, month, 1) <= today:
            effective_start_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
        elif action == "REVISION":
            effective_start_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            effective_start_date = f"{year}-{month:02d}-01"

        logger.info(f"📅 {action}: effective_start_date set to {effective_start_date}")

        _, days_in_month = calendar.monthrange(year, month)
        working_days = 0
        for day in range(1, days_in_month + 1):
            weekday = calendar.weekday(year, month, day)
            if weekday < 5:
                working_days += 1

        holiday_list = []
        if holidays:
            for holiday in holidays:
                holiday_list.append({
                    "date": holiday['date'],
                    "name": holiday['name'],
                    "type": holiday.get('type', 'PUBLIC'),
                    "strictness": holiday.get('strictness', 1)
                })

        if ui_config:
            max_cluster_switches = ui_config.get('max_cluster_switches', 3)
            min_same_cluster_visits = ui_config.get('min_same_cluster_visits', 2)
            route_order = ui_config.get('route_order', 'nearest')
            min_gap_days = ui_config.get('min_gap_days', 7)
            max_visits_without_revenue = ui_config.get('max_visits_without_revenue', 3)
            min_revenue_threshold = ui_config.get('min_revenue_threshold', 1000)
            visit_overage_cap = ui_config.get('visit_overage_cap', 1.2)
            growth_pct = ui_config.get('growth_pct', 0.15)
        else:
            max_cluster_switches = 3
            min_same_cluster_visits = 2
            route_order = 'nearest'
            min_gap_days = 7
            max_visits_without_revenue = 3
            min_revenue_threshold = 1000
            visit_overage_cap = 1.2
            growth_pct = 0.15 if tier_mix == "growth" else 0.10

        metadata = {
            "mr": mr_name,
            "month": month,
            "year": year,
            "territory": territory,
            "tier_mix": tier_mix,
            "effective_start_date": effective_start_date,
            "holidays": holiday_list,
            "channel_overrides": {},
            "doctor_kinds": {},
            "geo_overrides": {},
            "cluster_policy": {
                "max_cluster_switches_per_day": max_cluster_switches,
                "min_same_cluster_visits": min_same_cluster_visits,
                "route_order": route_order
            },
            "time_prefs": {},
            "daypart_caps": {"AM": 6, "PM": 4, "EVE": 2},
            "credit_blocks": {},
            "availability_alerts": {},
            "scheme_windows": [],
            "productivity": {
                "min_gap_days_same_customer": min_gap_days,
                "max_visits_without_revenue": max_visits_without_revenue,
                "conversion_window_days": 30,
                "min_revenue_threshold": min_revenue_threshold,
                "under_development_recency_days": 90,
                "visit_overage_ratio_cap": visit_overage_cap,
                "doctor_sample_followup_days": 14
            },
            "manager_overrides": {"blacklist_ids": [], "watchlist_ids": []},
            "territory_capacity": {territory: len(customers)},
            "profile": "standard",
            "growth_pct": growth_pct
        }

        return metadata

    async def generate_monthly_plan_with_clustering(self, mr_name: str, month: int, year: int,
                                                  customers: List[Dict[str, Any]],
                                                  action: str = "NEW_PLAN",
                                                  tier_mix: str = "balanced",
                                                  data_service: DataService = None,
                                                  enable_clustering: bool = True,
                                                  ui_config: Dict = None) -> Dict[str, Any]:
        """Generate plan with enhanced clustering support and UI configuration"""
        try:
            logger.info(f"🤖 Starting {action} with clustering for {mr_name}")

            self.data_service = data_service

            thread_id = None
            if action == "REVISION" and data_service:
                thread_id = self._get_existing_thread_id(data_service, mr_name, month, year)

            csv_data, id_mapping, clustering_metadata = self.prepare_customer_data_with_clustering(
                customers, action, month, year, enable_clustering
            )

            logger.info(f"🔍 Created ID mapping with {len(id_mapping)} entries")
            if len(id_mapping) > 0:
                sample_mappings = list(id_mapping.items())[:3]
                for db_id, customer_code in sample_mappings:
                    logger.info(f"  Sample mapping: {db_id} -> {customer_code}")

            location_info = {}
            if data_service:
                location_info = data_service.get_mr_location_info(mr_name)

            month_start = f"{year}-{month:02d}-01"
            last_day = calendar.monthrange(year, month)[1]
            month_end = f"{year}-{month:02d}-{last_day:02d}"

            holidays = []
            if data_service:
                holidays = data_service.get_holidays_for_period(
                    month_start, month_end,
                    location_info.get('state_code'),
                    location_info.get('district')
                )

            metadata = self.create_enhanced_ai_metadata(
                mr_name, month, year, customers, action, tier_mix, holidays, clustering_metadata, ui_config
            )

            ai_input = {
                "action": action,
                "compressed_data": csv_data,
                "metadata": metadata
            }

            logger.info(f"📤 Sending enhanced {action} request with clustering data")

            plan_result, used_thread_id = await self._call_openai_assistant_with_retry(
                ai_input, existing_thread_id=thread_id
            )

            logger.info("🔄 Mapping database IDs back to customer codes...")
            mapped_plan = self._map_ids_back_to_codes(plan_result, id_mapping)

            validated_plan = self._validate_ai_response(mapped_plan, customers)

            validated_plan['clustering_metadata'] = clustering_metadata
            validated_plan['thread_id'] = used_thread_id

            logger.info(f"🎉 Enhanced {action} completed with clustering and ID mapping for {mr_name}")
            return validated_plan

        except Exception as e:
            logger.error(f"❌ Enhanced {action} failed: {e}")
            raise

# ================================================================
# ENHANCED SYNCHRONOUS WRAPPER
# ================================================================

class EnhancedAIServiceSync:
    """Synchronous wrapper for enhanced AI service with clustering"""

    def __init__(self, config: Config):
        self.ai_service = EnhancedAIService(config)

    def generate_monthly_plan_with_clustering(self, mr_name: str, month: int, year: int,
                                            customers: List[Dict[str, Any]],
                                            action: str = "NEW_PLAN",
                                            tier_mix: str = "balanced",
                                            data_service: DataService = None,
                                            enable_clustering: bool = True,
                                            ui_config: Dict = None) -> Dict[str, Any]:
        """Synchronous wrapper for enhanced clustering with UI config"""
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.ai_service.generate_monthly_plan_with_clustering(
                                mr_name, month, year, customers, action, tier_mix,
                                data_service, enable_clustering, ui_config
                            )
                        )
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result(timeout=600)
            else:
                return loop.run_until_complete(
                    self.ai_service.generate_monthly_plan_with_clustering(
                        mr_name, month, year, customers, action, tier_mix,
                        data_service, enable_clustering, ui_config
                    )
                )

        except Exception as e:
            logger.error(f"❌ Enhanced AI call failed: {e}")
            raise

# ================================================================
# PROFESSIONAL PROGRESS TRACKER
# ================================================================

class ProfessionalProgressTracker:
    """Professional progress tracker for batch processing"""
    
    def __init__(self):
        self.tasks = {}
        self.widgets = None
        self.start_time = None

    def start_batch(self, mr_names: List[str]):
        """Start batch processing"""
        self.start_time = time.time()
        self.tasks = {mr_name: {
            'status': 'PENDING',
            'progress': 0,
            'message': 'Queued...',
            'customers': 0,
            'visits': 0,
            'revenue': 0
        } for mr_name in mr_names}
        
        self.progress_widgets = {}
        widget_list = []
        
        for mr_name in mr_names:
            progress_bar = widgets.IntProgress(
                value=0,
                min=0,
                max=100,
                description=f'{mr_name}:',
                bar_style='info',
                style={'bar_color': '#1f77b4', 'description_width': '150px'},
                layout=widgets.Layout(width='400px')
            )
            
            status_label = widgets.HTML(value='<span style="color: #666;">Queued...</span>')
            
            row = widgets.HBox([progress_bar, status_label])
            widget_list.append(row)
            
            self.progress_widgets[mr_name] = {
                'progress': progress_bar,
                'status': status_label
            }
        
        self.widgets = widgets.VBox(widget_list)

    def update_task(self, mr_name: str, status: str, progress: int, message: str, 
                   customers: int = 0, visits: int = 0, revenue: float = 0):
        """Update task progress"""
        if mr_name in self.tasks:
            self.tasks[mr_name].update({
                'status': status,
                'progress': progress,
                'message': message,
                'customers': customers,
                'visits': visits,
                'revenue': revenue
            })
            
            if mr_name in self.progress_widgets:
                self.progress_widgets[mr_name]['progress'].value = progress
                
                color_map = {
                    'PENDING': '#666',
                    'PROCESSING': '#1f77b4',
                    'COMPLETED': '#2ca02c',
                    'ERROR': '#d62728'
                }
                
                color = color_map.get(status, '#666')
                self.progress_widgets[mr_name]['status'].value = f'<span style="color: {color};">{message}</span>'
                
                if status == 'COMPLETED':
                    self.progress_widgets[mr_name]['progress'].bar_style = 'success'
                elif status == 'ERROR':
                    self.progress_widgets[mr_name]['progress'].bar_style = 'danger'
                elif status == 'PROCESSING':
                    self.progress_widgets[mr_name]['progress'].bar_style = 'info'

    def get_summary(self) -> Dict:
        """Get processing summary"""
        completed = sum(1 for task in self.tasks.values() if task['status'] == 'COMPLETED')
        errors = sum(1 for task in self.tasks.values() if task['status'] == 'ERROR')
        total_customers = sum(task['customers'] for task in self.tasks.values())
        total_visits = sum(task['visits'] for task in self.tasks.values())
        total_revenue = sum(task['revenue'] for task in self.tasks.values())
        
        success_rate = f"{(completed / len(self.tasks) * 100):.1f}%" if self.tasks else "0%"
        
        return {
            'completed': completed,
            'errors': errors,
            'total_customers': total_customers,
            'total_visits': total_visits,
            'total_revenue': total_revenue,
            'success_rate': success_rate
        }

# ================================================================
# PROFESSIONAL CSS STYLES
# ================================================================

PROFESSIONAL_CSS = """
<style>
.section-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    margin: 15px 0;
    border: 1px solid #e5e7eb;
}

.section-title {
    font-size: 1.4em;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 15px;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 8px;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.metric-value {
    font-size: 2em;
    font-weight: bold;
    margin-bottom: 5px;
}

.metric-label {
    font-size: 0.9em;
    opacity: 0.9;
}

.widget-box {
    background: #f8fafc;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}
</style>
"""

# ================================================================
# BASE PROFESSIONAL APP CLASS
# ================================================================

class ProfessionalAIVisitPlannerApp:
    """Base Professional AI Visit Planner Application"""
    
    def __init__(self):
        self.config = None
        self.data_service = None
        self.ai_service = None
        self.progress_tracker = None
        self.mr_data = []
        self.selected_mrs = []

    def load_medical_representatives(self):
        """Load medical representatives data"""
        try:
            self.mr_data = self.data_service.get_medical_representatives()
            logger.info(f"Loaded {len(self.mr_data)} medical representatives")
        except Exception as e:
            logger.error(f"Failed to load MRs: {e}")
            self.mr_data = []

    def create_professional_ui(self):
        """Create professional UI"""
        try:
            header = self._create_header()
            stats = self._create_statistics_section()
            config = self._create_config_section()
            selection = self._create_selection_section()
            action = self._create_action_section()
            
            main_ui = widgets.VBox([
                header,
                stats,
                config,
                selection,
                action
            ])
            
            return main_ui
            
        except Exception as e:
            logger.error(f"Failed to create UI: {e}")
            return widgets.HTML(f"<div style='color: red;'>Failed to create UI: {e}</div>")

    def _create_header(self):
        """Create header section"""
        return widgets.HTML('''
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em;">🚀 Enhanced AI Visit Planner</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">Professional Edition with Geographical Clustering</p>
        </div>
        ''')

    def _create_selection_section(self):
        """Create MR selection section"""
        self.mr_selector = widgets.SelectMultiple(
            options=[(f"{mr['name']} ({mr['territory']})", mr['name']) for mr in self.mr_data],
            value=[],
            description='Select MRs:',
            disabled=False,
            layout=widgets.Layout(width='100%', height='200px')
        )
        
        self.selection_info = widgets.HTML(
            value=f'<div style="background: #f0fdf4; padding: 10px; border-radius: 6px;">📋 Showing all {len(self.mr_data)} active MRs</div>'
        )
        
        def on_selection_change(change):
            selected_count = len(change['new'])
            self.process_button.description = f'🚀 Generate Enhanced AI Plans ({selected_count})'
            self.process_button.disabled = selected_count == 0
        
        self.mr_selector.observe(on_selection_change, names='value')
        
        return widgets.VBox([
            widgets.HTML('<div style="margin: 15px 0; font-weight: 500;">👥 Medical Representative Selection</div>'),
            self.mr_selector,
            self.selection_info
        ])

# ================================================================
# COMPLETE ENHANCED PROFESSIONAL APPLICATION WITH CLUSTERING
# ================================================================

class EnhancedProfessionalAIVisitPlannerApp(ProfessionalAIVisitPlannerApp):
    """Enhanced Professional AI Visit Planner with Clustering"""

    def __init__(self):
        # Initialize base application
        display(HTML(PROFESSIONAL_CSS))
        print("🚀 Initializing Enhanced AI Visit Planner with Clustering...")

        try:
            self.config = Config()
            self.data_service = DataService(self.config)
            self.ai_service = EnhancedAIServiceSync(self.config)
            self.progress_tracker = ProfessionalProgressTracker()

            self.mr_data = []
            self.selected_mrs = []

            print("✅ Enhanced application with clustering initialized successfully!")

        except Exception as e:
            print(f"❌ Failed to initialize enhanced application: {e}")
            raise

    def _create_config_section(self):
        """Enhanced configuration section with clustering and metadata options"""
        self.action_selector = widgets.ToggleButtons(
            options=[('🆕 New Plan', 'NEW_PLAN'), ('🔄 Revision', 'REVISION')],
            value='NEW_PLAN',
            description='Action Type:',
            style={'description_width': '120px', 'button_width': '150px'}
        )

        current_date = datetime.now()
        months = [(f"{i:02d} - {datetime(2024, i, 1).strftime('%B')}", i) for i in range(1, 13)]
        years = [(str(year), year) for year in range(2024, 2028)]

        self.month_selector = widgets.Dropdown(options=months, value=current_date.month, description='Month:')
        self.year_selector = widgets.Dropdown(options=years, value=current_date.year, description='Year:')
        self.tier_mix_selector = widgets.Dropdown(
            options=[('Balanced', 'balanced'), ('Growth', 'growth'), ('Maintenance', 'maintenance')],
            value='balanced',
            description='Strategy:'
        )

        # Enhanced clustering options
        self.enable_clustering = widgets.Checkbox(
            value=True,
            description='Enable Geo-Clustering',
            style={'description_width': 'initial'}
        )

        self.max_clusters_per_area = widgets.IntSlider(
            value=6,
            min=2,
            max=12,
            step=1,
            description='Max Clusters/Area:',
            style={'description_width': '140px'}
        )

        # New Metadata Configuration Options
        self.max_cluster_switches = widgets.IntSlider(
            value=3,
            min=1,
            max=6,
            step=1,
            description='Max Cluster Switches/Day:',
            style={'description_width': '180px'}
        )

        self.min_same_cluster_visits = widgets.IntSlider(
            value=2,
            min=1,
            max=5,
            step=1,
            description='Min Same Cluster Visits:',
            style={'description_width': '180px'}
        )

        self.route_order = widgets.Dropdown(
            options=[('Nearest First', 'nearest'), ('Manual Order', 'manual')],
            value='nearest',
            description='Route Order:',
            style={'description_width': '120px'}
        )

        self.min_gap_days = widgets.IntSlider(
            value=7,
            min=3,
            max=14,
            step=1,
            description='Min Gap Days (Same Customer):',
            style={'description_width': '200px'}
        )

        self.max_visits_without_revenue = widgets.IntSlider(
            value=3,
            min=1,
            max=6,
            step=1,
            description='Max Visits w/o Revenue:',
            style={'description_width': '180px'}
        )

        self.min_revenue_threshold = widgets.IntSlider(
            value=1000,
            min=500,
            max=5000,
            step=250,
            description='Min Revenue Threshold (₹):',
            style={'description_width': '200px'}
        )

        self.visit_overage_cap = widgets.FloatSlider(
            value=1.2,
            min=1.0,
            max=2.0,
            step=0.1,
            description='Visit Overage Cap:',
            style={'description_width': '150px'}
        )

        self.growth_percentage = widgets.FloatSlider(
            value=0.15,
            min=0.05,
            max=0.30,
            step=0.05,
            description='Growth Target %:',
            style={'description_width': '150px'}
        )

        date_row = widgets.HBox([self.month_selector, self.year_selector, self.tier_mix_selector])
        clustering_row = widgets.HBox([self.enable_clustering, self.max_clusters_per_area])

        cluster_policy_row1 = widgets.HBox([self.max_cluster_switches, self.min_same_cluster_visits])
        cluster_policy_row2 = widgets.HBox([self.route_order])

        productivity_row1 = widgets.HBox([self.min_gap_days, self.max_visits_without_revenue])
        productivity_row2 = widgets.HBox([self.min_revenue_threshold, self.visit_overage_cap])

        growth_row = widgets.HBox([self.growth_percentage])

        def on_action_change(change):
            """Update MR list based on action"""
            action = change['new']
            month = self.month_selector.value
            year = self.year_selector.value

            if action == "REVISION":
                available_mrs = self.data_service.get_mrs_with_plans(month, year)
                self.mr_selector.options = [(f"{mr['name']} ({mr['territory']})", mr['name']) for mr in available_mrs]
                self.selection_info.value = f'<div style="background: #eff6ff; padding: 10px; border-radius: 6px;">📋 Showing {len(available_mrs)} MRs with existing plans</div>'
            else:
                self.mr_selector.options = [(f"{mr['name']} ({mr['territory']})", mr['name']) for mr in self.mr_data]
                self.selection_info.value = f'<div style="background: #f0fdf4; padding: 10px; border-radius: 6px;">📋 Showing all {len(self.mr_data)} active MRs</div>'

        self.action_selector.observe(on_action_change, names='value')

        clustering_info = widgets.HTML('''
        <div style="background: #f0f9ff; padding: 12px; border-radius: 8px; margin: 10px 0;">
            <strong>🗺️ Geographical Clustering:</strong><br>
            • Groups customers by location for optimal route planning<br>
            • Reduces travel time and increases visit efficiency<br>
            • Automatically balances cluster sizes within each area
        </div>
        ''')

        metadata_info = widgets.HTML('''
        <div style="background: #fefce8; padding: 12px; border-radius: 8px; margin: 10px 0;">
            <strong>⚙️ Advanced Configuration:</strong><br>
            • Customize AI planning behavior and constraints<br>
            • Set productivity thresholds and visit policies<br>
            • Configure growth targets and revenue expectations
        </div>
        ''')

        return widgets.VBox([
            self.action_selector,
            widgets.HTML('<div style="margin: 15px 0; font-weight: 500;">📅 Planning Period & Strategy</div>'),
            date_row,
            widgets.HTML('<div style="margin: 15px 0; font-weight: 500;">🗺️ Clustering Options</div>'),
            clustering_row,
            clustering_info,
            widgets.HTML('<div style="margin: 15px 0; font-weight: 500;">🎯 Cluster Policy Configuration</div>'),
            cluster_policy_row1,
            cluster_policy_row2,
            widgets.HTML('<div style="margin: 15px 0; font-weight: 500;">📊 Productivity Configuration</div>'),
            productivity_row1,
            productivity_row2,
            widgets.HTML('<div style="margin: 15px 0; font-weight: 500;">📈 Growth Configuration</div>'),
            growth_row,
            metadata_info
        ])

    def _create_statistics_section(self):
        """Enhanced statistics with clustering info"""
        territories = {}
        for mr in self.mr_data:
            territory = mr.get('territory', 'Unknown')
            territories[territory] = territories.get(territory, 0) + 1

        territory_list = sorted(territories.items(), key=lambda x: x[1], reverse=True)[:8]
        territory_html = '<br>'.join([f'• {territory}: {count}' for territory, count in territory_list])

        stats_html = f'''
        <div class="section-card">
            <div class="section-title">📊 Enhanced System Status</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 15px; margin: 20px 0;">
                <div class="metric-card">
                    <div class="metric-value">{len(self.mr_data)}</div>
                    <div class="metric-label">Active MRs</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(territories)}</div>
                    <div class="metric-label">Territories</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">🗺️ Enabled</div>
                    <div class="metric-label">Clustering</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">Ready</div>
                    <div class="metric-label">System</div>
                </div>
            </div>
            <div style="margin-top: 15px; font-size: 0.9rem;">
                <strong>Top Territories:</strong><br>{territory_html}
            </div>
            <div style="margin-top: 15px; padding: 10px; background: #ecfdf5; border-radius: 6px;">
                <strong>🚀 New Features:</strong> Geographical clustering, area-based grouping,
                route optimization, and enhanced travel efficiency metrics.
            </div>
        </div>
        '''

        return widgets.HTML(stats_html)

    def _verify_customer_codes_in_plan(self, plan_result: Dict[str, Any], customers: List[Dict[str, Any]]):
        """Verify that all customer codes in the plan are valid"""
        try:
            valid_codes = set(c.get('customer_code', '') for c in customers if c.get('customer_code'))
            
            plan_codes = set()
            if 'daily_plan' in plan_result:
                for day in plan_result['daily_plan']:
                    if 'dayparts' in day:
                        for daypart, customer_ids in day['dayparts'].items():
                            if isinstance(customer_ids, list):
                                plan_codes.update(str(cid) for cid in customer_ids)
            
            invalid_codes = plan_codes - valid_codes
            if invalid_codes:
                logger.warning(f"⚠️ Found {len(invalid_codes)} invalid customer codes in plan: {invalid_codes}")
            else:
                logger.info(f"✅ All {len(plan_codes)} customer codes in plan are valid")
                
        except Exception as e:
            logger.error(f"❌ Customer code verification failed: {e}")

    def _process_single_mr_enhanced(self, mr_name: str, month: int, year: int,
                                  action: str, tier_mix: str, enable_clustering: bool = True,
                                  max_clusters: int = 6) -> bool:
        """Enhanced MR processing with clustering support and proper ID mapping"""
        thread_id = None
        try:
            self.progress_tracker.update_task(
                mr_name=mr_name,
                status='PROCESSING',
                progress=20,
                message='Loading customers...'
            )

            customers = self.data_service.get_customer_data(mr_name)

            if not customers:
                self.progress_tracker.update_task(
                    mr_name=mr_name,
                    status='ERROR',
                    progress=0,
                    message='No customers found'
                )
                return False

            logger.info(f"📊 Loaded {len(customers)} customers for {mr_name}")
            if customers:
                sample_customer = customers[0]
                logger.info(f"Sample customer: ID={sample_customer.get('id')}, Code={sample_customer.get('customer_code')}")

            if enable_clustering:
                self.progress_tracker.update_task(
                    mr_name=mr_name,
                    status='PROCESSING',
                    progress=35,
                    message=f'🗺️ Analyzing {len(customers)} customers with clustering...'
                )
            else:
                self.progress_tracker.update_task(
                    mr_name=mr_name,
                    status='PROCESSING',
                    progress=40,
                    message=f'Analyzing {len(customers)} customers...'
                )

            self.progress_tracker.update_task(
                mr_name=mr_name,
                status='PROCESSING',
                progress=70,
                message=f'🤖 Calling Enhanced AI for {action.upper()}...'
            )

            ui_config = {
                'max_cluster_switches': self.max_cluster_switches.value,
                'min_same_cluster_visits': self.min_same_cluster_visits.value,
                'route_order': self.route_order.value,
                'min_gap_days': self.min_gap_days.value,
                'max_visits_without_revenue': self.max_visits_without_revenue.value,
                'min_revenue_threshold': self.min_revenue_threshold.value,
                'visit_overage_cap': self.visit_overage_cap.value,
                'growth_pct': self.growth_percentage.value
            }

            plan_result = self.ai_service.generate_monthly_plan_with_clustering(
                mr_name, month, year, customers, action, tier_mix,
                self.data_service, enable_clustering, ui_config
            )

            self.progress_tracker.update_task(
                mr_name=mr_name,
                status='PROCESSING',
                progress=90,
                message='Saving enhanced plan to database...'
            )

            thread_id = plan_result.get('thread_id')
            clustering_metadata = plan_result.get('clustering_metadata', {})
            clean_plan_result = {k: v for k, v in plan_result.items()
                               if k not in ['thread_id', 'clustering_metadata']}

            logger.info("🔍 Verifying customer codes in final plan...")
            self._verify_customer_codes_in_plan(clean_plan_result, customers)

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

            saved_plan = self.data_service.save_monthly_plan(plan_data)

            visits = clean_plan_result['executive_summary']['planned_total_visits']
            revenue = clean_plan_result['executive_summary']['expected_revenue']
            thread_display = thread_id[:8] + "..." if thread_id else "None"

            clusters_info = ""
            if clustering_metadata.get('clustering_enabled'):
                total_clusters = clustering_metadata.get('total_clusters', 0)
                total_areas = clustering_metadata.get('total_areas', 0)
                clusters_info = f"🗺️ {total_clusters}C/{total_areas}A"

            success_message = f'✅ Enhanced {action} completed! {clusters_info} Thread: {thread_display}'

            self.progress_tracker.update_task(
                mr_name=mr_name,
                status='COMPLETED',
                progress=100,
                message=success_message,
                customers=len(customers),
                visits=visits,
                revenue=revenue
            )

            return True

        except Exception as e:
            error_msg = f"Enhanced processing failed: {str(e)}"
            if thread_id:
                error_msg += f" (Thread: {thread_id[:8]}...)"

            self.progress_tracker.update_task(
                mr_name=mr_name,
                status='ERROR',
                progress=0,
                message=error_msg
            )
            logger.error(f"❌ Enhanced processing error for {mr_name}: {e}")
            return False

    def _process_plans_professional_enhanced(self):
        """Enhanced processing with clustering options"""
        try:
            month = self.month_selector.value
            year = self.year_selector.value
            action = self.action_selector.value
            tier_mix = self.tier_mix_selector.value
            enable_clustering = self.enable_clustering.value
            max_clusters = self.max_clusters_per_area.value

            clustering_status = "🗺️ WITH CLUSTERING" if enable_clustering else "📍 WITHOUT CLUSTERING"

            print(f"\n🚀 Starting Enhanced {action} processing {clustering_status}")
            print(f"📅 {month:02d}/{year} | Strategy: {tier_mix} | {len(self.selected_mrs)} MRs")
            if enable_clustering:
                print(f"🗺️ Max {max_clusters} clusters per area")

            self.progress_tracker.start_batch(self.selected_mrs)
            display(self.progress_tracker.widgets)

            results = {'completed': 0, 'errors': 0, 'total_customers': 0, 'total_visits': 0, 'total_revenue': 0}

            for i, mr_name in enumerate(self.selected_mrs):
                try:
                    success = self._process_single_mr_enhanced(
                        mr_name, month, year, action, tier_mix, enable_clustering, max_clusters
                    )

                    if success:
                        results['completed'] += 1
                    else:
                        results['errors'] += 1

                    time.sleep(0.5)

                except Exception as e:
                    results['errors'] += 1
                    self.progress_tracker.update_task(mr_name, 'ERROR', 0, f'Enhanced processing failed: {str(e)}')

            self._display_enhanced_final_summary(results, enable_clustering)

        except Exception as e:
            print(f"❌ Enhanced batch processing failed: {e}")
        finally:
            self.process_button.disabled = False
            self.process_button.description = f'🚀 Generate Enhanced AI Plans ({len(self.selected_mrs)})'

    def _display_enhanced_final_summary(self, results: Dict, clustering_enabled: bool):
        """Enhanced final summary with clustering metrics"""
        summary = self.progress_tracker.get_summary()

        clustering_badge = "🗺️ Clustering Enabled" if clustering_enabled else "📍 Standard Mode"

        summary_html = f'''
        <div class="section-card" style="margin: 20px 0;">
            <h3 style="text-align: center; color: #1f2937;">📊 Enhanced Processing Complete!</h3>
            <div style="text-align: center; margin: 10px 0;">
                <span style="background: #{'059669' if clustering_enabled else '6b7280'}; color: white;
                           padding: 6px 12px; border-radius: 20px; font-size: 0.9rem;">
                    {clustering_badge}
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0;">
                <div class="metric-card">
                    <div class="metric-value">{summary["success_rate"]}</div>
                    <div class="metric-label">Success Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary["total_customers"]}</div>
                    <div class="metric-label">Customers</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary["total_visits"]}</div>
                    <div class="metric-label">Visits</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">₹{summary["total_revenue"]:,.0f}</div>
                    <div class="metric-label">Revenue</div>
                </div>
            </div>
            <div style="text-align: center; color: #6b7280;">
                ✅ {summary["completed"]} completed • ❌ {summary["errors"]} errors
            </div>
            {f'<div style="text-align: center; margin-top: 15px; padding: 10px; background: #ecfdf5; border-radius: 6px;"><strong>🗺️ Enhanced with geographical clustering</strong><br>Optimized routes and travel efficiency included in all plans</div>' if clustering_enabled else ''}
        </div>
        '''

        display(HTML(summary_html))
        print(f"🎉 Successfully generated {summary['completed']} enhanced plans with clustering!")

    def _start_processing(self, button):
        """Enhanced processing starter"""
        self.selected_mrs = list(self.mr_selector.value)

        if not self.selected_mrs:
            display(HTML('<div style="color: red; text-align: center;">❌ No MRs selected!</div>'))
            return

        self.process_button.disabled = True
        self.process_button.description = '⏳ Processing Enhanced Plans...'

        self._process_plans_professional_enhanced()

    def _create_action_section(self):
        """Enhanced action section"""
        self.process_button = widgets.Button(
            description='🚀 Generate Enhanced AI Plans',
            button_style='success',
            icon='rocket',
            disabled=True,
            layout=widgets.Layout(width='280px', height='50px', margin='20px')
        )

        self.process_button.on_click(self._start_processing)
        return self.process_button

    def _create_selection_section(self):
        """Create MR selection section"""
        self.mr_selector = widgets.SelectMultiple(
            options=[(f"{mr['name']} ({mr['territory']})", mr['name']) for mr in self.mr_data],
            value=[],
            description='Select MRs:',
            disabled=False,
            layout=widgets.Layout(width='100%', height='200px')
        )
        
        self.selection_info = widgets.HTML(
            value=f'<div style="background: #f0fdf4; padding: 10px; border-radius: 6px;">📋 Showing all {len(self.mr_data)} active MRs</div>'
        )
        
        def on_selection_change(change):
            selected_count = len(change['new'])
            self.process_button.description = f'🚀 Generate Enhanced AI Plans ({selected_count})'
            self.process_button.disabled = selected_count == 0
        
        self.mr_selector.observe(on_selection_change, names='value')
        
        return widgets.VBox([
            widgets.HTML('<div style="margin: 15px 0; font-weight: 500;">👥 Medical Representative Selection</div>'),
            self.mr_selector,
            self.selection_info
        ])

# ================================================================
# ENVIRONMENT SETUP AND LAUNCH
# ================================================================

def setup_environment():
    """Setup environment for different platforms"""
    try:
        import google.colab
        in_colab = True
        print("📍 Detected Google Colab environment")
        
        from google.colab import userdata
        try:
            os.environ['SUPABASE_URL'] = userdata.get('SUPABASE_URL')
            os.environ['SUPABASE_ANON_KEY'] = userdata.get('SUPABASE_ANON_KEY')
            os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
            os.environ['OPENAI_ASSISTANT_ID'] = userdata.get('OPENAI_ASSISTANT_ID')
            print("✅ Environment variables loaded from Colab secrets")
        except Exception as e:
            print(f"❌ Failed to load from Colab secrets: {e}")
            print("Please add your credentials to Colab secrets")
            return False
            
    except ImportError:
        print("📍 Detected local environment")
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ Environment variables loaded from .env file")
        except ImportError:
            print("💡 Install python-dotenv for .env file support: pip install python-dotenv")
    
    return True

def validate_environment() -> bool:
    """Validate required environment variables"""
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_ANON_KEY', 
        'OPENAI_API_KEY',
        'OPENAI_ASSISTANT_ID'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\n🔧 Setup Instructions:")
        print("1. For Google Colab: Add variables to Secrets (🔑 icon)")
        print("2. For local: Create .env file with your credentials")
        print("\nRequired variables:")
        for var in required_vars:
            print(f"   {var}=your-{var.lower().replace('_', '-')}-here")
        return False
    
    print("✅ All environment variables are set")
    return True

def test_database_connection() -> bool:
    """Test database connection"""
    try:
        config = Config()
        response = config.supabase.table('medical_representatives').select('count', count='exact').limit(1).execute()
        print(f"✅ Database connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def launch_enhanced_production_app():
    """Launch complete enhanced production application"""
    try:
        print("🔧 Setting up environment...")
        if not setup_environment():
            return

        print("🔍 Validating environment...")
        if not validate_environment():
            return

        print("🔍 Testing database connection...")
        if not test_database_connection():
            return

        print("🚀 Launching Enhanced Professional AI Visit Planner with Clustering...")
        app = EnhancedProfessionalAIVisitPlannerApp()
        app.load_medical_representatives()

        if app.mr_data:
            ui = app.create_professional_ui()
            clear_output(wait=True)
            display(ui)
            print("✅ Enhanced Production AI Visit Planner with Clustering Ready!")
            print("🗺️ Features: Geographical clustering, proper ID mapping, route optimization")
            print("🎯 New: Advanced configuration options and real-time progress tracking")
        else:
            print("❌ No medical representatives found")

    except Exception as e:
        print(f"❌ Enhanced launch failed: {e}")
        import traceback
        traceback.print_exc()

# ================================================================
# AUTO-LAUNCH
# ================================================================

if __name__ == "__main__":
    launch_enhanced_production_app()

# For Jupyter/Colab environments
try:
    if hasattr(__builtins__, '__IPYTHON__'):
        launch_enhanced_production_app()
except:
    pass
