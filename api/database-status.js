const { ensureSchema, ensureWorkspace, getWorkspaceId, isDatabaseConfigured, query } = require("./lib/db");

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
    const workspaceId = getWorkspaceId(request);
    await ensureSchema();
    await ensureWorkspace(workspaceId);
    const result = await query("SELECT COUNT(*)::int AS batches FROM batches WHERE workspace_id = $1", [workspaceId]);
    return response.status(200).json({
      mode: "database",
      connected: true,
      workspace_id: workspaceId,
      batches: result.rows[0].batches,
      message: "Postgres persistence is active.",
    });
  } catch (error) {
    return response.status(500).json({
      mode: "browser",
      connected: false,
      error: error.message,
      message: "Database connection failed; browser-only workflow remains available.",
    });
  }
};
