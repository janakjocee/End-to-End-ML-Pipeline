"""
Model Retraining DAG
====================

Apache Airflow DAG for automated model retraining pipeline.
Triggered by drift detection or on a schedule.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.http_operator import SimpleHttpOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.http_sensor import HttpSensor
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable

# Default arguments
default_args = {
    'owner': 'ml-platform',
    'depends_on_past': False,
    'email': ['ml-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# DAG configuration
dag = DAG(
    'model_retraining_pipeline',
    default_args=default_args,
    description='Automated model retraining pipeline',
    schedule_interval=timedelta(days=1),  # Run daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'retraining', 'automation'],
    max_active_runs=1,
)

# Service URLs
DATA_SERVICE_URL = os.getenv('DATA_SERVICE_URL', 'http://data-service:8000')
FEATURE_SERVICE_URL = os.getenv('FEATURE_SERVICE_URL', 'http://feature-service:8000')
TRAINING_SERVICE_URL = os.getenv('TRAINING_SERVICE_URL', 'http://training-service:8000')
MODEL_REGISTRY_URL = os.getenv('MODEL_REGISTRY_URL', 'http://model-registry:8000')
MONITORING_SERVICE_URL = os.getenv('MONITORING_SERVICE_URL', 'http://monitoring-service:8000')
INFERENCE_API_URL = os.getenv('INFERENCE_API_URL', 'http://inference-api:8000')


# ============================================
# Task Functions
# ============================================

def check_drift_status(**context):
    """Check if drift detection triggered retraining."""
    # Check for recent drift reports
    response = requests.get(
        f"{MONITORING_SERVICE_URL}/reports",
        params={'limit': 10}
    )
    
    if response.status_code == 200:
        reports = response.json()
        
        # Check for recent drift detections
        recent_drift = any(
            r['drift_detected'] and 
            (datetime.utcnow() - datetime.fromisoformat(r['created_at'].replace('Z', '+00:00'))).days < 1
            for r in reports
        )
        
        if recent_drift:
            context['ti'].xcom_push(key='retraining_reason', value='drift_detected')
            return 'trigger_retraining'
    
    # Check if scheduled retraining
    scheduled_retraining = Variable.get("scheduled_retraining", default_var=False)
    if scheduled_retraining:
        context['ti'].xcom_push(key='retraining_reason', value='scheduled')
        return 'trigger_retraining'
    
    return 'skip_retraining'


def prepare_training_data(**context):
    """Prepare data for training."""
    model_name = context['dag_run'].conf.get('model_name', 'churn-predictor')
    
    # Get latest dataset
    response = requests.get(
        f"{DATA_SERVICE_URL}/datasets",
        params={'dataset_name': 'churn_data', 'limit': 1}
    )
    
    if response.status_code != 200:
        raise Exception("Failed to get dataset info")
    
    datasets = response.json()
    if not datasets:
        raise Exception("No datasets found")
    
    latest_dataset = datasets[0]
    
    # Apply feature engineering
    feature_response = requests.post(
        f"{FEATURE_SERVICE_URL}/engineer",
        json={
            'dataset_name': latest_dataset['dataset_name'],
            'dataset_version': latest_dataset['version'],
            'feature_set_name': 'churn_features_v1',
            'output_name': f'{model_name}_training_data'
        }
    )
    
    if feature_response.status_code != 200:
        raise Exception("Feature engineering failed")
    
    result = feature_response.json()
    
    context['ti'].xcom_push(key='training_dataset', value=result['dataset_name'])
    context['ti'].xcom_push(key='training_version', value=result['version'])
    
    return f"Prepared training data: {result['dataset_name']} v{result['version']}"


def train_new_model(**context):
    """Train a new model."""
    model_name = context['dag_run'].conf.get('model_name', 'churn-predictor')
    model_type = context['dag_run'].conf.get('model_type', 'xgboost')
    
    training_dataset = context['ti'].xcom_pull(key='training_dataset')
    training_version = context['ti'].xcom_pull(key='training_version')
    
    # Start training
    response = requests.post(
        f"{TRAINING_SERVICE_URL}/train/async",
        json={
            'experiment_name': f'{model_name}_retraining_{datetime.utcnow().strftime("%Y%m%d")}',
            'model_name': f'{model_name}_v2',
            'model_type': model_type,
            'dataset_name': training_dataset,
            'dataset_version': training_version,
            'target_column': 'churn',
            'hyperparameter_tuning': True,
            'n_trials': 50
        }
    )
    
    if response.status_code != 200:
        raise Exception("Training request failed")
    
    result = response.json()
    run_id = result['run_id']
    
    context['ti'].xcom_push(key='training_run_id', value=run_id)
    
    return f"Training started with run_id: {run_id}"


def wait_for_training(**context):
    """Wait for training to complete."""
    import time
    
    run_id = context['ti'].xcom_pull(key='training_run_id')
    
    max_attempts = 60  # 60 minutes
    for attempt in range(max_attempts):
        response = requests.get(
            f"{TRAINING_SERVICE_URL}/train/{run_id}/status"
        )
        
        if response.status_code == 200:
            status = response.json()
            
            if status['status'] == 'completed':
                context['ti'].xcom_push(key='training_result', value=status)
                return 'training_complete'
            elif status['status'] == 'failed':
                raise Exception(f"Training failed: {status.get('error', 'Unknown error')}")
        
        time.sleep(60)  # Wait 1 minute
    
    raise Exception("Training timeout")


def evaluate_model(**context):
    """Evaluate trained model against current production model."""
    training_result = context['ti'].xcom_pull(key='training_result')
    
    if not training_result or 'result' not in training_result:
        raise Exception("Training result not available")
    
    new_metrics = training_result['result']['metrics']
    
    # Get current production model metrics
    model_name = context['dag_run'].conf.get('model_name', 'churn-predictor')
    
    response = requests.get(
        f"{MODEL_REGISTRY_URL}/models/{model_name}/production"
    )
    
    if response.status_code == 200:
        current_model = response.json()
        current_metrics = current_model.get('metrics', {})
        
        # Compare metrics
        comparison = {
            'new_f1': new_metrics.get('f1_score', 0),
            'current_f1': current_metrics.get('f1_score', 0),
            'improvement': new_metrics.get('f1_score', 0) - current_metrics.get('f1_score', 0)
        }
        
        context['ti'].xcom_push(key='model_comparison', value=comparison)
        
        # Decide whether to promote
        if comparison['improvement'] > 0.01:  # 1% improvement threshold
            return 'promote_model'
    
    return 'discard_model'


def promote_to_production(**context):
    """Promote new model to production."""
    training_result = context['ti'].xcom_pull(key='training_result')
    
    if not training_result or 'result' not in training_result:
        raise Exception("Training result not available")
    
    model_name = training_result['result']['model_name']
    
    # Register model
    response = requests.post(
        f"{MODEL_REGISTRY_URL}/register",
        json={
            'model_name': model_name,
            'run_id': training_result['result']['run_id'],
            'description': f'Automatically retrained on {datetime.utcnow().isoformat()}'
        }
    )
    
    if response.status_code != 200:
        raise Exception("Model registration failed")
    
    registered = response.json()
    version = registered['version']
    
    # Promote to production
    promote_response = requests.post(
        f"{MODEL_REGISTRY_URL}/transition",
        json={
            'model_name': model_name,
            'model_version': version,
            'stage': 'Production',
            'comment': 'Auto-promoted after retraining pipeline'
        }
    )
    
    if promote_response.status_code != 200:
        raise Exception("Model promotion failed")
    
    # Update inference API
    reload_response = requests.post(
        f"{INFERENCE_API_URL}/reload"
    )
    
    return f"Model {model_name} v{version} promoted to production"


def send_notification(**context):
    """Send notification about retraining completion."""
    retraining_reason = context['ti'].xcom_pull(key='retraining_reason')
    model_comparison = context['ti'].xcom_pull(key='model_comparison')
    
    message = {
        'event': 'retraining_completed',
        'timestamp': datetime.utcnow().isoformat(),
        'reason': retraining_reason,
        'comparison': model_comparison,
        'dag_id': context['dag'].dag_id,
        'run_id': context['run_id']
    }
    
    # Send to webhook if configured
    webhook_url = Variable.get("retraining_webhook_url", default_var=None)
    if webhook_url:
        try:
            requests.post(webhook_url, json=message)
        except Exception as e:
            print(f"Failed to send notification: {e}")
    
    return "Notification sent"


# ============================================
# DAG Tasks
# ============================================

# Start task
start = DummyOperator(
    task_id='start_retraining_pipeline',
    dag=dag,
)

# Check drift status
check_drift = BranchPythonOperator(
    task_id='check_drift_status',
    python_callable=check_drift_status,
    dag=dag,
)

# Skip retraining
skip_retraining = DummyOperator(
    task_id='skip_retraining',
    dag=dag,
)

# Prepare data
prepare_data = PythonOperator(
    task_id='prepare_training_data',
    python_callable=prepare_training_data,
    dag=dag,
)

# Train model
train_model = PythonOperator(
    task_id='train_new_model',
    python_callable=train_new_model,
    dag=dag,
)

# Wait for training
wait_training = BranchPythonOperator(
    task_id='wait_for_training',
    python_callable=wait_for_training,
    dag=dag,
)

# Training failed
training_failed = DummyOperator(
    task_id='training_failed',
    dag=dag,
)

# Evaluate model
evaluate = BranchPythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

# Promote model
promote = PythonOperator(
    task_id='promote_to_production',
    python_callable=promote_to_production,
    dag=dag,
)

# Discard model
discard = DummyOperator(
    task_id='discard_model',
    dag=dag,
)

# Send notification
notify = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    trigger_rule=TriggerRule.ONE_SUCCESS,
    dag=dag,
)

# End task
end = DummyOperator(
    task_id='end_pipeline',
    trigger_rule=TriggerRule.ONE_SUCCESS,
    dag=dag,
)

# ============================================
# Task Dependencies
# ============================================

start >> check_drift
check_drift >> [prepare_data, skip_retraining]
prepare_data >> train_model >> wait_training
wait_training >> [evaluate, training_failed]
evaluate >> [promote, discard]
promote >> notify >> end
discard >> notify
training_failed >> notify
skip_retraining >> end