const crypto = require("node:crypto");
const { Pool } = require("pg");
const { schema } = require("./schema");

let pool;
let schemaReady = false;

function isDatabaseConfigured() {
  return Boolean(process.env.DATABASE_URL || process.env.POSTGRES_URL);
}

function getPool() {
  if (!isDatabaseConfigured()) return null;
  if (!pool) {
    pool = new Pool({
      connectionString: process.env.DATABASE_URL || process.env.POSTGRES_URL,
      ssl: process.env.POSTGRES_SSL === "false" ? false : { rejectUnauthorized: false },
      max: 3,
    });
  }
  return pool;
}

async function query(text, params = []) {
  const database = getPool();
  if (!database) {
    const error = new Error("DATABASE_URL is not configured.");
    error.code = "DATABASE_NOT_CONFIGURED";
    throw error;
  }
  await ensureSchema();
  return database.query(text, params);
}

async function transaction(callback) {
  const database = getPool();
  if (!database) {
    const error = new Error("DATABASE_URL is not configured.");
    error.code = "DATABASE_NOT_CONFIGURED";
    throw error;
  }
  await ensureSchema();
  const client = await database.connect();
  try {
    await client.query("BEGIN");
    const result = await callback((text, params = []) => client.query(text, params));
    await client.query("COMMIT");
    return result;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
}

async function rawQuery(text, params = []) {
  const database = getPool();
  if (!database) {
    const error = new Error("DATABASE_URL is not configured.");
    error.code = "DATABASE_NOT_CONFIGURED";
    throw error;
  }
  return database.query(text, params);
}

async function ensureSchema() {
  if (schemaReady) return;
  await rawQuery(schema);
  schemaReady = true;
}

async function ensureWorkspace(workspaceId, name = "Demo workspace") {
  await query(
    `
      INSERT INTO workspaces (id, name)
      VALUES ($1, $2)
      ON CONFLICT (id) DO NOTHING
    `,
    [workspaceId, name],
  );
}

async function writeAudit({ workspaceId, actor, action, entityType, entityId, metadata = {} }) {
  await query(
    `
      INSERT INTO audit_logs (id, workspace_id, actor, action, entity_type, entity_id, metadata)
      VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
    `,
    [crypto.randomUUID(), workspaceId, actor, action, entityType, entityId, JSON.stringify(metadata)],
  );
}

module.exports = {
  ensureSchema,
  ensureWorkspace,
  isDatabaseConfigured,
  query,
  transaction,
  writeAudit,
};
