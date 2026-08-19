/**
 * Vercel Blob client-upload handler.
 *
 * Browser → Vercel Blob (not through Python body) → bypass ~4.5 MB limit.
 * Requires BLOB_READ_WRITE_TOKEN on the deployment.
 *
 * Set BLOB_ACCESS=private if your store is Private (default: public).
 */
const { handleUpload } = require("@vercel/blob/client");

const MAX_BYTES = 100 * 1024 * 1024; // 100 MB
const BLOB_ACCESS =
  (process.env.BLOB_ACCESS || "public").toLowerCase() === "private"
    ? "private"
    : "public";

module.exports = async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, X-StudyQuiz-Pin");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "POST, OPTIONS");
    return res.status(405).json({ error: "Method not allowed" });
  }

  const token = process.env.BLOB_READ_WRITE_TOKEN || "";
  if (!token) {
    return res.status(503).json({
      error:
        "BLOB_READ_WRITE_TOKEN is not set on this deployment. " +
        "Add it under Vercel → Settings → Environment Variables (Production + Preview) and Redeploy.",
    });
  }

  let body = req.body;
  if (typeof body === "string") {
    try {
      body = JSON.parse(body);
    } catch {
      return res.status(400).json({ error: "Invalid JSON body" });
    }
  }
  if (!body || typeof body !== "object") {
    return res.status(400).json({ error: "Missing JSON body for handleUpload" });
  }

  try {
    const jsonResponse = await handleUpload({
      body,
      request: req,
      token,
      onBeforeGenerateToken: async (_pathname) => ({
        // Do not restrict MIME types — Safari often sends empty/odd types for PDFs.
        // Restricting content types caused hard-to-debug 400s on real devices.
        maximumSizeInBytes: MAX_BYTES,
        addRandomSuffix: true,
        tokenPayload: JSON.stringify({
          purpose: "studyquiz-document",
          access: BLOB_ACCESS,
        }),
      }),
      onUploadCompleted: async ({ blob }) => {
        console.log("blob upload completed", blob && blob.url);
      },
    });
    // Tell the browser which access mode to send on PUT.
    if (jsonResponse && jsonResponse.type === "blob.generate-client-token") {
      return res.status(200).json({ ...jsonResponse, access: BLOB_ACCESS });
    }
    return res.status(200).json(jsonResponse);
  } catch (error) {
    console.error("blob-upload error", error);
    return res.status(400).json({
      error: (error && error.message) || "Blob upload handler failed",
    });
  }
};
