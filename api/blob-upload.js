/** Private, PIN-authorized client uploads. SDK verifies completion callbacks. */
const { handleUpload } = require('@vercel/blob/client');
const { timingSafeEqual } = require('node:crypto');

function validPin(headers) {
  const expected = (process.env.STUDYQUIZ_ACCESS_PIN || process.env.ACCESS_PIN || process.env.STUDYQUIZ_API_SECRET || '').trim();
  if (!expected || (process.env.VERCEL && expected.length < 16)) return false; // Uploads always fail closed, including misconfigured previews.
  const supplied = String(headers['x-studyquiz-pin'] || (headers.authorization || '').replace(/^Bearer /i, '')).trim();
  const a = Buffer.from(expected), b = Buffer.from(supplied);
  return a.length === b.length && timingSafeEqual(a, b);
}

module.exports = async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  if (req.method !== 'POST') return res.status(405).json({error: 'POST required'});
  let body;
  try { body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body; }
  catch { return res.status(400).json({error: 'Invalid JSON'}); }
  if (!body || typeof body !== 'object') return res.status(400).json({error: 'Invalid body'});
  // Completion requests are authorized by handleUpload's signature validation;
  // never require a student's PIN on the storage provider's callback.
  if (body.type !== 'blob.upload-completed' && !validPin(req.headers)) {
    return res.status(401).json({error: 'Access PIN required'});
  }
  if (body.type !== 'blob.upload-completed' && body.type !== 'blob.generate-client-token') {
    return res.status(400).json({error: 'Unsupported upload event'});
  }
  const token = process.env.BLOB_READ_WRITE_TOKEN;
  if (!token || (process.env.BLOB_ACCESS || 'private') !== 'private') {
    return res.status(503).json({error: 'A private Blob store must be configured'});
  }
  try {
    const result = await handleUpload({
      body, request: req, token,
      onBeforeGenerateToken: async (pathname) => {
        if (!validPin(req.headers)) throw new Error('Access PIN required');
        if (!/^uploads\/[a-f0-9]{32}-[A-Za-z0-9_.()+-]{1,180}$/.test(pathname) ||
            !/\.(pdf|docx|pptx|txt|md|csv)$/i.test(pathname)) throw new Error('Unsupported document pathname');
        // Use the deployment's configured URL, never an untrusted Host header.
        const origin = process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : 'http://127.0.0.1:8000';
        const permit = await fetch(`${origin}/api/blob/permit`, {
          method: 'POST', redirect: 'error', signal: AbortSignal.timeout(15000),
          headers: {'content-type': 'application/json', 'x-studyquiz-pin': process.env.STUDYQUIZ_ACCESS_PIN || process.env.ACCESS_PIN || process.env.STUDYQUIZ_API_SECRET,
                    cookie: req.headers.cookie || ''},
          body: JSON.stringify({pathname}),
        });
        if (!permit.ok) throw new Error(permit.status === 429 ? 'Daily upload allowance reached' : 'Could not authorize private upload');
        return {
          callbackUrl: `${origin}/api/blob/upload`,
          maximumSizeInBytes: Math.min(100, Math.max(1, Number(process.env.MAX_UPLOAD_MB) || 100)) * 1024 * 1024,
          addRandomSuffix: false, // Browser pathname already contains 128 random bits and is recorded in the DB.
          allowOverwrite: false,
          cacheControlMaxAge: 60,
          validUntil: Date.now() + 15 * 60 * 1000,
          tokenPayload: JSON.stringify({purpose: 'studyquiz-document', pathname}),
        };
      },
      onUploadCompleted: async () => {}, // No private URLs/tokens written to logs.
    });
    return res.status(200).json({...result, access: 'private'});
  } catch (error) {
    const limited = String(error && error.message).includes('allowance');
    return res.status(limited ? 429 : 400).json({error: limited ? 'Daily upload allowance reached' : 'Private upload authorization failed. Check the PIN and deployment configuration.'});
  }
};
