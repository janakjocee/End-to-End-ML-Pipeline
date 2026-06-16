const { scoreBatch } = require("./scoring");

module.exports = (request, response) => {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    return response.status(405).json({ error: "Use POST with a JSON body containing a records array." });
  }

  const records = request.body?.records;
  if (!Array.isArray(records)) {
    return response.status(400).json({ error: "Expected JSON body: { \"records\": [...] }" });
  }
  if (records.length > 1000) {
    return response.status(413).json({ error: "Batch limit is 1000 records per request." });
  }

  return response.status(200).json(scoreBatch(records));
};
