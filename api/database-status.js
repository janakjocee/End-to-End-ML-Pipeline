const { getAuthContext, sendAuthError } = require("./lib/auth");
const { ensureSchema, ensureWorkspace, isDatabaseConfigured, query } = require("./lib/db");

module.exports = async (request, response) => {
  if (request.method !== "GET") {
    response.setHeader("Allow", "GET");
    return response.status(405).json({ error: "Use GET." });
  }

  if (!isDatabaseConfigured()) {
    return response.status(200).json({
      mode: "browser",
      connected: false,
      message: "DATABASE_URL is not configured. The app is using browser-only demo history.",
    });
  }

  try {
    const auth = getAuthContext(request);
    const workspaceId = auth.workspace_id;
    await ensureSchema();
    await ensureWorkspace(workspaceId);
    const result = await query("SELECT COUNT(*)::int AS batches FROM batches WHERE workspace_id = $1", [workspaceId]);
    return response.status(200).json({
      mode: "database",
      connected: true,
      auth_required: auth.auth_required,
      workspace_id: workspaceId,
      batches: result.rows[0].batches,
      message: "Postgres persistence is active.",
    });
  } catch (error) {
    if (error.statusCode === 401) return sendAuthError(response, error);
    return response.status(500).json({
      mode: "browser",
      connected: false,
      error: error.message,
      message: "Database connection failed; browser-only workflow remains available.",
    });
  }
};
