"""
Sample Dataset Generator
========================

Generate sample data for demonstrating the ML Platform.
Creates a customer churn dataset with realistic features.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_churn_dataset(
    n_samples: int = 10000,
    random_state: int = 42,
    drift: bool = False
) -> pd.DataFrame:
    """
    Generate a customer churn dataset.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        drift: If True, generate data with drift for testing
        
    Returns:
        DataFrame with customer data and churn labels
    """
    np.random.seed(random_state)
    
    # Customer demographics
    customer_id = [f'CUST_{i:06d}' for i in range(n_samples)]
    
    # Tenure (months as customer)
    tenure = np.random.exponential(24, n_samples).clip(1, 72).astype(int)
    
    # Monthly charges
    base_charge = np.random.normal(65, 30, n_samples).clip(20, 120)
    if drift:
        # Add drift by increasing charges
        base_charge = base_charge * 1.3
    monthly_charges = base_charge.round(2)
    
    # Total charges
    total_charges = (monthly_charges * tenure * np.random.uniform(0.9, 1.1, n_samples)).round(2)
    
    # Contract type
    contract_probs = [0.55, 0.25, 0.20]  # Month-to-month, One year, Two year
    if drift:
        contract_probs = [0.70, 0.20, 0.10]  # More month-to-month
    contract = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'],
        n_samples,
        p=contract_probs
    )
    
    # Payment method
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        n_samples,
        p=[0.35, 0.20, 0.25, 0.20]
    )
    
    # Internet service
    internet_service = np.random.choice(
        ['DSL', 'Fiber optic', 'No'],
        n_samples,
        p=[0.45, 0.40, 0.15]
    )
    
    # Additional services
    online_security = np.where(
        internet_service == 'No',
        'No internet service',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65])
    )
    
    online_backup = np.where(
        internet_service == 'No',
        'No internet service',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.40, 0.60])
    )
    
    device_protection = np.where(
        internet_service == 'No',
        'No internet service',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65])
    )
    
    tech_support = np.where(
        internet_service == 'No',
        'No internet service',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
    )
    
    streaming_tv = np.where(
        internet_service == 'No',
        'No internet service',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.45, 0.55])
    )
    
    streaming_movies = np.where(
        internet_service == 'No',
        'No internet service',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.45, 0.55])
    )
    
    # Phone service
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
    
    # Multiple lines
    multiple_lines = np.where(
        phone_service == 'No',
        'No phone service',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.45, 0.55])
    )
    
    # Gender
    gender = np.random.choice(['Male', 'Female'], n_samples)
    
    # Senior citizen
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    
    # Partner
    partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.50, 0.50])
    
    # Dependents
    dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
    
    # Paperless billing
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.60, 0.40])
    
    # Generate churn based on features
    churn_probability = np.zeros(n_samples)
    
    # Higher churn for month-to-month contracts
    churn_probability += (contract == 'Month-to-month') * 0.25
    
    # Higher churn for electronic check payments
    churn_probability += (payment_method == 'Electronic check') * 0.15
    
    # Higher churn for fiber optic (often more expensive)
    churn_probability += (internet_service == 'Fiber optic') * 0.10
    
    # Lower churn for longer tenure
    churn_probability -= (tenure > 24) * 0.15
    churn_probability -= (tenure > 48) * 0.10
    
    # Lower churn with tech support
    churn_probability -= (tech_support == 'Yes') * 0.10
    
    # Lower churn with online security
    churn_probability -= (online_security == 'Yes') * 0.08
    
    # Higher churn for senior citizens
    churn_probability += senior_citizen * 0.05
    
    # Higher churn for higher monthly charges
    churn_probability += (monthly_charges > 80) * 0.10
    
    # Normalize probability
    churn_probability = np.clip(churn_probability, 0.05, 0.95)
    
    if drift:
        # Increase churn rate for drift scenario
        churn_probability = np.clip(churn_probability * 1.5, 0, 0.95)
    
    # Generate churn labels
    churn = np.random.binomial(1, churn_probability)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_id,
        'gender': gender,
        'senior_citizen': senior_citizen,
        'partner': partner,
        'dependents': dependents,
        'tenure': tenure,
        'phone_service': phone_service,
        'multiple_lines': multiple_lines,
        'internet_service': internet_service,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'contract': contract,
        'paperless_billing': paperless_billing,
        'payment_method': payment_method,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'churn': churn
    })
    
    return df


def save_dataset(df: pd.DataFrame, output_dir: str, name: str, version: str):
    """Save dataset to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f'{name}_{version}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Save as Parquet
    parquet_path = os.path.join(output_dir, f'{name}_{version}.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet: {parquet_path}")
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Samples: {len(df):,}")
    print(f"  Features: {len(df.columns)}")
    print(f"  Churn Rate: {df['churn'].mean():.2%}")
    print(f"  Avg Monthly Charges: ${df['monthly_charges'].mean():.2f}")
    print(f"  Avg Tenure: {df['tenure'].mean():.1f} months")


def main():
    parser = argparse.ArgumentParser(description='Generate sample churn dataset')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--output-dir', type=str, default='datasets', help='Output directory')
    parser.add_argument('--name', type=str, default='churn_data', help='Dataset name')
    parser.add_argument('--version', type=str, default='v1', help='Dataset version')
    parser.add_argument('--drift', action='store_true', help='Generate data with drift')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Generating {args.samples:,} samples...")
    df = generate_churn_dataset(
        n_samples=args.samples,
        random_state=args.seed,
        drift=args.drift
    )
    
    save_dataset(df, args.output_dir, args.name, args.version)
    
    # Also generate drift dataset if requested
    if not args.drift:
        print("\nGenerating drift dataset...")
        df_drift = generate_churn_dataset(
            n_samples=args.samples,
            random_state=args.seed + 1,
            drift=True
        )
        save_dataset(df_drift, args.output_dir, args.name, 'v1_drift')


if __name__ == '__main__':
    main()