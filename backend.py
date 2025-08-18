#!/usr/bin/env python3
"""
Clean Backend for AI Visit Planner - Streamlit Compatible
========================================================
Backend logic without Jupyter dependencies
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
    os.system("pip install supabase openai pandas nest-asyncio scikit-learn geopy")
    from supabase import create_client, Client
    import openai
    from openai import OpenAI
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import geopy.distance
    import nest_asyncio

# Apply nest_asyncio for async compatibility
def setup_asyncio():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            nest_asyncio.apply()
    except RuntimeError:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except:
            pass

setup_asyncio()

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

    def login(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user against the user_profiles table."""
        try:
            response = self.supabase.table('user_profiles')\
                .select('*')\
                .eq('email', email)\
                .eq('password', password)\
                .single()\
                .execute()

            if response.data:
                logger.info(f"User {email} logged in successfully.")
                return response.data
            else:
                logger.warning(f"Login failed for user {email}.")
                return None
        except Exception as e:
            logger.error(f"Error during login for user {email}: {e}")
            return None

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
    """Base AI service with enhanced JSON parsing"""
    
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

    def _extract_json_from_markdown(self, content: str) -> str:
        """Extract JSON content from markdown code blocks"""
        try:
            import re
            
            patterns = [
                r'```json\s*(.*?)\s*```',  # ```json ... ```
                r'```\s*(.*?)\s*```',      # ``` ... ```
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    extracted = match.group(1).strip()
                    logger.info(f"📋 Extracted JSON from markdown")
                    return extracted
            
            # Look for content between first { and last }
            first_brace = content.find('{')
            last_brace = content.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                extracted = content[first_brace:last_brace + 1]
                logger.info("📋 Extracted JSON by finding braces")
                return extracted
            
            logger.warning("⚠️ No JSON extraction pattern matched, returning original")
            return content
            
        except Exception as e:
            logger.error(f"❌ JSON extraction failed: {e}")
            return content

    def _manual_json_cleanup(self, content: str) -> str:
        """Manual JSON cleanup as last resort"""
        try:
            cleaned = content.strip()
            cleaned = cleaned.replace('```json', '').replace('```', '')
            cleaned = cleaned.strip()
            
            if not cleaned.startswith('{'):
                start_idx = cleaned.find('{')
                if start_idx != -1:
                    cleaned = cleaned[start_idx:]
            
            if not cleaned.endswith('}'):
                end_idx = cleaned.rfind('}')
                if end_idx != -1:
                    cleaned = cleaned[:end_idx + 1]
            
            return cleaned
            
        except Exception as e:
            logger.error(f"❌ Manual cleanup failed: {e}")
            return content

    async def _call_openai_assistant_with_retry(self, ai_input: Dict, existing_thread_id: str = None) -> tuple[Dict, str]:
        """Call OpenAI assistant with retry logic and enhanced JSON parsing"""
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
                
                # ENHANCED JSON PARSING
                try:
                    # Direct JSON parsing
                    plan_result = json.loads(response_content)
                    logger.info("✅ Direct JSON parsing successful")
                    return plan_result, thread_id
                except json.JSONDecodeError:
                    # Try markdown extraction
                    logger.info("🔄 Trying markdown extraction...")
                    cleaned_content = self._extract_json_from_markdown(response_content)
                    try:
                        plan_result = json.loads(cleaned_content)
                        logger.info("✅ Markdown extraction successful")
                        return plan_result, thread_id
                    except json.JSONDecodeError:
                        # Try manual cleanup
                        logger.info("🔄 Trying manual cleanup...")
                        manual_cleaned = self._manual_json_cleanup(response_content)
                        try:
                            plan_result = json.loads(manual_cleaned)
                            logger.info("✅ Manual cleanup successful")
                            return plan_result, thread_id
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ All parsing methods failed: {e}")
                            logger.error(f"Content preview: {response_content[:300]}")
                            raise Exception(f"Invalid JSON response from AI after all cleanup attempts: {response_content[:200]}...")
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
