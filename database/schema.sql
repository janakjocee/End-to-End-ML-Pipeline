CREATE TABLE IF NOT EXISTS workspaces (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS batches (
  id UUID PRIMARY KEY,
  workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  label TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'csv_upload',
  row_count INTEGER NOT NULL,
  summary JSONB NOT NULL,
  mapping JSONB NOT NULL DEFAULT '{}'::JSONB,
  drift_report JSONB NOT NULL DEFAULT '{}'::JSONB,
  outcome TEXT NOT NULL DEFAULT 'Pending review',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
  id UUID PRIMARY KEY,
  batch_id UUID NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
  workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  customer_id TEXT NOT NULL,
  features JSONB NOT NULL,
  score JSONB NOT NULL,
  action_status TEXT NOT NULL DEFAULT 'Not started',
  outcome TEXT NOT NULL DEFAULT 'Pending',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS action_outcomes (
  id UUID PRIMARY KEY,
  prediction_id UUID REFERENCES predictions(id) ON DELETE SET NULL,
  batch_id UUID NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
  workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  status TEXT NOT NULL,
  outcome TEXT,
  notes TEXT,
  saved_revenue NUMERIC,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_versions (
  id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'candidate',
  metrics JSONB NOT NULL DEFAULT '{}'::JSONB,
  threshold_policy JSONB NOT NULL DEFAULT '{}'::JSONB,
  artifact_uri TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS drift_reports (
  id UUID PRIMARY KEY,
  workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  batch_id UUID REFERENCES batches(id) ON DELETE SET NULL,
  report JSONB NOT NULL,
  severity TEXT NOT NULL DEFAULT 'info',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_logs (
  id UUID PRIMARY KEY,
  workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  actor TEXT NOT NULL DEFAULT 'demo-user',
  action TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS batches_workspace_created_idx ON batches (workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS predictions_batch_idx ON predictions (batch_id);
CREATE INDEX IF NOT EXISTS predictions_workspace_risk_idx ON predictions (workspace_id, ((score->>'risk_band')));
CREATE INDEX IF NOT EXISTS action_outcomes_batch_idx ON action_outcomes (batch_id);
CREATE INDEX IF NOT EXISTS drift_reports_workspace_created_idx ON drift_reports (workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS audit_logs_workspace_created_idx ON audit_logs (workspace_id, created_at DESC);
