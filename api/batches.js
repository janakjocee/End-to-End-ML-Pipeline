const {
  ensureWorkspace,
  isDatabaseConfigured,
  query,
  transaction,
  writeAudit,
} = require("./lib/db");
const { getAuthContext, sendAuthError } = require("./lib/auth");
const crypto = require("node:crypto");

function unavailable(response) {
  return response.status(200).json({
    mode: "browser",
    connected: false,
    batches: [],
    message: "DATABASE_URL is not configured. Using browser-only demo history.",
  });
}

async function listBatches(request, response) {
  if (!isDatabaseConfigured()) return unavailable(response);
  const auth = getAuthContext(request);
  const workspaceId = auth.workspace_id;
  await ensureWorkspace(workspaceId);
  const result = await query(
    `
      SELECT id, label, source, row_count, summary, mapping, drift_report, outcome, created_at, updated_at
      FROM batches
      WHERE workspace_id = $1
      ORDER BY created_at DESC
      LIMIT 25
    `,
    [workspaceId],
  );
  return response.status(200).json({
    mode: "database",
    connected: true,
    auth_required: auth.auth_required,
    workspace_id: workspaceId,
    batches: result.rows.map((row) => ({
      id: row.id,
      label: row.label,
      source: row.source,
      row_count: row.row_count,
      summary: row.summary,
      mapping: row.mapping,
      drift_report: row.drift_report,
      outcome: row.outcome,
      created_at: row.created_at,
      updated_at: row.updated_at,
    })),
  });
}

async function createBatch(request, response) {
  if (!isDatabaseConfigured()) return unavailable(response);
  const auth = getAuthContext(request);
  const workspaceId = auth.workspace_id;
  const actor = auth.actor;
  const body = request.body || {};
  const rows = Array.isArray(body.rows) ? body.rows.slice(0, 500) : [];
  if (!body.id || !body.label || !body.summary || !rows.length) {
    return response.status(400).json({ error: "Expected id, label, summary, and non-empty rows." });
  }

  await ensureWorkspace(workspaceId);
  await transaction(async (tx) => {
    await tx(
      `
        INSERT INTO batches (id, workspace_id, label, source, row_count, summary, mapping, drift_report, outcome, created_at)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb, $9, $10)
        ON CONFLICT (id) DO UPDATE SET
          label = EXCLUDED.label,
          row_count = EXCLUDED.row_count,
          summary = EXCLUDED.summary,
          mapping = EXCLUDED.mapping,
          drift_report = EXCLUDED.drift_report,
          outcome = EXCLUDED.outcome,
          updated_at = NOW()
      `,
      [
        body.id,
        workspaceId,
        body.label,
        body.source || "csv_upload",
        body.row_count || rows.length,
        JSON.stringify(body.summary),
        JSON.stringify(body.mapping || {}),
        JSON.stringify(body.drift_report || {}),
        body.outcome || "Pending review",
        body.created_at || new Date().toISOString(),
      ],
    );
    await tx("DELETE FROM predictions WHERE batch_id = $1 AND workspace_id = $2", [body.id, workspaceId]);
    for (const row of rows) {
      await tx(
        `
          INSERT INTO predictions (
            id, batch_id, workspace_id, customer_id, features, score, action_status, outcome
          )
          VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8)
        `,
        [
          crypto.randomUUID(),
          body.id,
          workspaceId,
          String(row.customer_id || "unknown"),
          JSON.stringify(row.features || {}),
          JSON.stringify(row.score || {}),
          row.action_status || "Not started",
          row.outcome || "Pending",
        ],
      );
    }
  });
  await writeAudit({
    workspaceId,
    actor,
    action: "batch_saved",
    entityType: "batch",
    entityId: body.id,
    metadata: { label: body.label, row_count: rows.length },
  });

  return response.status(201).json({ mode: "database", connected: true, id: body.id, workspace_id: workspaceId });
}

async function updateBatch(request, response) {
  if (!isDatabaseConfigured()) return unavailable(response);
  const auth = getAuthContext(request);
  const workspaceId = auth.workspace_id;
  const actor = auth.actor;
  const { id, outcome } = request.body || {};
  if (!id || !outcome) return response.status(400).json({ error: "Expected id and outcome." });
  await ensureWorkspace(workspaceId);
  const result = await query(
    `
      UPDATE batches
      SET outcome = $1, updated_at = NOW()
      WHERE id = $2 AND workspace_id = $3
      RETURNING id, label, outcome, updated_at
    `,
    [outcome, id, workspaceId],
  );
  if (!result.rowCount) return response.status(404).json({ error: "Batch not found." });
  await writeAudit({
    workspaceId,
    actor,
    action: "batch_outcome_updated",
    entityType: "batch",
    entityId: id,
    metadata: { outcome },
  });
  return response.status(200).json({ mode: "database", connected: true, batch: result.rows[0] });
}

module.exports = async (request, response) => {
  try {
    if (request.method === "GET") return listBatches(request, response);
    if (request.method === "POST") return createBatch(request, response);
    if (request.method === "PATCH") return updateBatch(request, response);
    response.setHeader("Allow", "GET, POST, PATCH");
    return response.status(405).json({ error: "Use GET, POST, or PATCH." });
  } catch (error) {
    if (error.statusCode === 401) return sendAuthError(response, error);
    return response.status(500).json({
      mode: "browser",
      connected: false,
      error: error.message,
      message: "Database operation failed; browser-only workflow remains available.",
    });
  }
};
