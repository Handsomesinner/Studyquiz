/**
 * Vercel Blob client-upload handler.
 *
 * Browser uploads go Browser → Vercel Blob (not through the Python function
 * body), which bypasses the ~4.5 MB serverless request limit.
 *
 * Requires BLOB_READ_WRITE_TOKEN in the Vercel project environment.
 */
const { handleUpload } = require("@vercel/blob/client");

const MAX_BYTES = 100 * 1024 * 1024; // 100 MB

const ALLOWED = [
  "application/pdf",
  "application/msword",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.ms-powerpoint",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  "text/plain",
  "text/markdown",
  "text/csv",
  "text/html",
  "application/octet-stream", // some browsers send this for .docx/.pdf
];

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Method not allowed" });
  }

  // Vercel creates this when you connect Blob with "Add a read-write token" ticked.
  const token = process.env.BLOB_READ_WRITE_TOKEN || "";
  if (!token) {
    return res.status(503).json({
      error:
        "BLOB_READ_WRITE_TOKEN is not set on this deployment. " +
        "Fix: Vercel → Project → Settings → Environment Variables → add " +
        "BLOB_READ_WRITE_TOKEN (from Storage → Blob → .env.local / token). " +
        "Apply to Production, then Redeploy. " +
        `Hint: BLOB_STORE_ID is ${process.env.BLOB_STORE_ID ? "set" : "also missing"}.`,
    });
  }

  // Vercel Node may leave body as object or string depending on config.
  const body = typeof req.body === "string" ? JSON.parse(req.body) : req.body;

  try {
    const jsonResponse = await handleUpload({
      body,
      request: req,
      onBeforeGenerateToken: async (_pathname /*, clientPayload */) => ({
        allowedContentTypes: ALLOWED,
        maximumSizeInBytes: MAX_BYTES,
        addRandomSuffix: true,
        tokenPayload: JSON.stringify({ purpose: "studyquiz-document" }),
      }),
      onUploadCompleted: async ({ blob }) => {
        // Optional hook — indexing is triggered by the client via /api/documents.
        console.log("blob upload completed", blob?.url);
      },
    });
    return res.status(200).json(jsonResponse);
  } catch (error) {
    console.error("blob-upload error", error);
    return res.status(400).json({
      error: (error && error.message) || "Blob upload handler failed",
    });
  }
};
