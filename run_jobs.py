import argparse
import logging
from datetime import datetime
from backend import Config, DataService, EnhancedAIServiceSync

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _process_mr_plan(ai_service, data_service, mr, month, year, action, tier_mix='balanced'):
    """Helper function to process a plan for a single MR."""
    try:
        mr_name = mr['name']
        logger.info(f"Processing {action} for {mr_name} for {month}/{year}")
        customers = data_service.get_customer_data(mr_name)
        if not customers:
            logger.warning(f"No customers found for {mr_name}, skipping.")
            return

        plan_result = ai_service.generate_monthly_plan_with_clustering(
            mr_name, month, year, customers, action, tier_mix,
            data_service, enable_clustering=True
        )

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
        logger.info(f"Successfully processed and saved {action} for {mr_name}")

    except Exception as e:
        logger.error(f"Error processing {action} for {mr['name']}: {e}")

def run_new_plan_job():
    """Job to create new plans for MRs without a plan for the current month."""
    logger.info("Running new plan job...")
    try:
        config = Config()
        data_service = DataService(config)
        ai_service = EnhancedAIServiceSync(config)

        now = datetime.now()
        month, year = now.month, now.year

        all_mrs = data_service.get_medical_representatives()
        mrs_with_plans = data_service.get_mrs_with_plans(month, year)
        mrs_with_plans_names = {mr['name'] for mr in mrs_with_plans}

        mrs_to_plan = [mr for mr in all_mrs if mr['name'] not in mrs_with_plans_names]

        logger.info(f"Found {len(mrs_to_plan)} MRs for new plan generation.")

        for mr in mrs_to_plan:
            _process_mr_plan(ai_service, data_service, mr, month, year, 'NEW_PLAN')

    except Exception as e:
        logger.error(f"New plan job failed: {e}")
    logger.info("New plan job finished.")

def run_revision_job():
    """Job to create revisions for MRs with an existing plan for the current month."""
    logger.info("Running revision job...")
    try:
        config = Config()
        data_service = DataService(config)
        ai_service = EnhancedAIServiceSync(config)

        now = datetime.now()
        month, year = now.month, now.year

        mrs_to_revise = data_service.get_mrs_with_plans(month, year)
        logger.info(f"Found {len(mrs_to_revise)} MRs for revision.")

        for mr in mrs_to_revise:
            _process_mr_plan(ai_service, data_service, mr, month, year, 'REVISION')

    except Exception as e:
        logger.error(f"Revision job failed: {e}")
    logger.info("Revision job finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automated planner jobs.")
    parser.add_argument("job_type", choices=["new_plan", "revision"], help="The type of job to run.")

    args = parser.parse_args()

    if args.job_type == "new_plan":
        run_new_plan_job()
    elif args.job_type == "revision":
        run_revision_job()
