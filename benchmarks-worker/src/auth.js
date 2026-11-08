// Trusted-key auth. Two valid credentials, both short-lived or secret:
//   1. HMAC token minted by GET /api/token (form path) — `exp.signature`
//      where signature = HMAC-SHA256(TOKEN_SECRET, exp). 10-min TTL.
//   2. Static Bearer SUBMIT_BEARER (CLI / `oflm bench --submit`, out of band).
// The web form never sees a static secret; bots without a token can't POST.

async function hmac(secret, msg) {
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]);
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(msg));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function mintToken(secret, ttlS = 600) {
  const exp = Math.floor(Date.now() / 1000) + ttlS;
  return `${exp}.${await hmac(secret, String(exp))}`;
}

export async function isAuthorized(req, env) {
  const token = req.headers.get("X-OFLM-Token") || "";
  if (token) {
    const [exp, sig] = token.split(".");
    if (!exp || !sig || !env.TOKEN_SECRET) return false;
    if (Number(exp) < Math.floor(Date.now() / 1000)) return false;
    const expect = await hmac(env.TOKEN_SECRET, exp);
    // constant-time compare (lengths equal for hex HMACs of same algorithm)
    let diff = 0;
    for (let i = 0; i < Math.max(expect.length, sig.length); i++)
      diff |= (expect.charCodeAt(i) || 0) ^ (sig.charCodeAt(i) || 0);
    return diff === 0;
  }
  const auth = req.headers.get("Authorization") || "";
  return Boolean(env.SUBMIT_BEARER && auth === `Bearer ${env.SUBMIT_BEARER}`);
}
