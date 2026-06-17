function getHeader(request, name) {
  return request.headers[name] || request.headers[name.toLowerCase()];
}

function authRequired() {
  return Boolean(process.env.WORKFLOW_API_KEY || process.env.REQUIRE_WORKFLOW_AUTH === "true");
}

function getBearerToken(request) {
  const authorization = getHeader(request, "authorization") || "";
  if (authorization.toLowerCase().startsWith("bearer ")) return authorization.slice(7).trim();
  return getHeader(request, "x-workflow-token") || "";
}

function getAuthContext(request) {
  const required = authRequired();
  const expected = process.env.WORKFLOW_API_KEY || "";
  const token = getBearerToken(request);
  if (required && (!expected || token !== expected)) {
    const error = new Error("Persistence API authentication failed.");
    error.statusCode = 401;
    throw error;
  }

  return {
    actor: getHeader(request, "x-user-email") || "demo-user",
    auth_required: required,
    authenticated: required ? token === expected : false,
    workspace_id: getHeader(request, "x-workspace-id") || process.env.DEFAULT_WORKSPACE_ID || "demo-workspace",
  };
}

function sendAuthError(response, error) {
  return response.status(error.statusCode || 401).json({
    connected: false,
    error: error.message,
    mode: "auth_required",
    message: "Set the workflow API token in the app access controls to use protected persistence APIs.",
  });
}

module.exports = {
  authRequired,
  getAuthContext,
  sendAuthError,
};
