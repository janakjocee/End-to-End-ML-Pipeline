# AWS Deployment Guide

This guide covers deploying the ML Platform on AWS infrastructure.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          AWS CLOUD                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Route 53  │    │  CloudFront │    │     WAF     │         │
│  │    (DNS)    │    │    (CDN)    │    │  (Security) │         │
│  └──────┬──────┘    └──────┬──────┘    └─────────────┘         │
│         │                    │                                   │
│  ┌──────▼────────────────────▼──────┐                          │
│  │      Application Load Balancer   │                          │
│  └──────┬────────────────────┬──────┘                          │
│         │                    │                                   │
│  ┌──────▼──────┐    ┌────────▼──────┐                          │
│  │    ECS      │    │     EKS       │                          │
│  │  (Fargate)  │    │ (Kubernetes)  │                          │
│  └──────┬──────┘    └───────┬───────┘                          │
│         │                    │                                   │
│  ┌──────▼────────────────────▼──────┐                          │
│  │         Service Mesh (App Mesh)   │                          │
│  └──────┬────────────────────┬──────┘                          │
│         │                    │                                   │
│  ┌──────▼──────┐    ┌────────▼──────┐    ┌─────────────┐       │
│  │   RDS       │    │ DocumentDB    │    │ ElastiCache │       │
│  │(PostgreSQL) │    │  (MongoDB)    │    │   (Redis)   │       │
│  └─────────────┘    └───────────────┘    └─────────────┘       │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │     S3      │    │    ECR      │    │ CloudWatch  │         │
│  │ (Artifacts) │    │ (Images)    │    │  (Metrics)  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- AWS CLI configured
- Terraform installed
- kubectl installed
- Docker installed

## Deployment Options

### Option 1: ECS Fargate (Recommended for Simplicity)

```bash
# Deploy using Terraform
cd terraform/aws/ecs

# Initialize
terraform init

# Plan
terraform plan -var="environment=production"

# Apply
terraform apply -var="environment=production"
```

### Option 2: EKS (Recommended for Scale)

```bash
# Create EKS cluster
cd terraform/aws/eks

terraform init
terraform apply

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name ml-platform

# Deploy applications
kubectl apply -f k8s/aws/
```

## Service Configuration

### RDS PostgreSQL

```yaml
# Database configuration
engine: postgres
engine_version: "15.4"
instance_class: db.r5.xlarge
allocated_storage: 100
multi_az: true
backup_retention_period: 30
```

### DocumentDB

```yaml
# MongoDB-compatible
engine: docdb
instance_class: db.r5.large
cluster_size: 3
backup_retention_period: 30
```

### ElastiCache Redis

```yaml
# Redis cluster
engine: redis
node_type: cache.r5.large
num_cache_nodes: 2
automatic_failover_enabled: true
```

### S3 Buckets

```bash
# Create buckets
aws s3 mb s3://ml-platform-artifacts-prod
aws s3 mb s3://ml-platform-datasets-prod
aws s3 mb s3://ml-platform-mlflow-prod

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket ml-platform-artifacts-prod \
  --versioning-configuration Status=Enabled
```

## Networking

### VPC Configuration

```yaml
vpc:
  cidr: 10.0.0.0/16
  azs: 3
  private_subnets: 3
  public_subnets: 3
  enable_nat_gateway: true
  enable_vpn_gateway: false
```

### Security Groups

| Service | Ingress | Egress |
|---------|---------|--------|
| ALB | 80, 443 (0.0.0.0/0) | All |
| ECS Tasks | All (VPC only) | All |
| RDS | 5432 (ECS only) | None |
| DocumentDB | 27017 (ECS only) | None |
| ElastiCache | 6379 (ECS only) | None |

## IAM Roles

### ECS Task Role

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::ml-platform-*",
        "arn:aws:s3:::ml-platform-*/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/ecs/ml-platform:*"
    }
  ]
}
```

## Cost Estimation

### Monthly Costs (Production)

| Service | Instance | Cost/Month |
|---------|----------|------------|
| ECS Fargate | 4 vCPU, 8GB x 6 | ~$400 |
| RDS | db.r5.xlarge | ~$350 |
| DocumentDB | db.r5.large x 3 | ~$600 |
| ElastiCache | cache.r5.large x 2 | ~$200 |
| ALB | - | ~$25 |
| S3 | 500GB | ~$15 |
| Data Transfer | 1TB | ~$90 |
| **Total** | | **~$1,680** |

## Monitoring

### CloudWatch Dashboards

```bash
# Create dashboard
aws cloudwatch put-dashboard \
  --dashboard-name MLPlatform-Production \
  --dashboard-body file://cloudwatch-dashboard.json
```

### CloudWatch Alarms

| Metric | Threshold | Action |
|--------|-----------|--------|
| CPU Utilization | > 80% | Scale up |
| Memory Utilization | > 80% | Scale up |
| Error Rate | > 5% | Page on-call |
| Latency p95 | > 500ms | Alert |

## Backup and Recovery

### Automated Backups

```bash
# RDS backup window
aws rds modify-db-instance \
  --db-instance-identifier ml-platform-postgres \
  --preferred-backup-window 03:00-04:00 \
  --backup-retention-period 30

# DocumentDB backup
aws docdb create-db-cluster \
  --db-cluster-identifier ml-platform-docdb \
  --backup-retention-period 30 \
  --preferred-backup-window 04:00-05:00
```

### Disaster Recovery

1. **Cross-Region Replication**: Enable for S3 buckets
2. **Read Replicas**: Create for RDS in secondary region
3. **Backup Testing**: Monthly restore drills

## Security Best Practices

1. **Encryption**: Enable at rest and in transit
2. **VPC Endpoints**: Use for AWS service access
3. **Secrets Manager**: Store credentials securely
4. **WAF**: Protect against common attacks
5. **Security Groups**: Principle of least privilege

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| High latency | Resource constraints | Scale ECS tasks |
| Connection errors | Security groups | Verify rules |
| Out of memory | Large models | Increase memory |
| Slow training | Insufficient CPU | Use larger instances |

### Logs

```bash
# ECS logs
aws logs tail /ecs/ml-platform --follow

# RDS logs
aws rds describe-db-log-files \
  --db-instance-identifier ml-platform-postgres
```

## Cleanup

```bash
# Destroy infrastructure
terraform destroy

# Delete S3 buckets
aws s3 rb s3://ml-platform-artifacts-prod --force
aws s3 rb s3://ml-platform-datasets-prod --force
```

---

For support, contact: ml-platform-team@company.com