const { scoreCustomer } = require("./scoring");

module.exports = (request, response) => {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    return response.status(405).json({ error: "Use POST with a JSON body." });
  }
  const features = request.body?.features || request.body || {};
  return response.status(200).json(scoreCustomer(features));
};
