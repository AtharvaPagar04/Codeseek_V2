# Authentication and Credentials

## Authentication Paths

CodeSeek accepts either:

- A GitHub-backed application session in the `codeseek_session` cookie.
- `Authorization: Bearer <CODESEEK_API_KEY>` when the server API key is configured.

GitHub access can be established through OAuth or direct token submission. The backend validates the token with GitHub, persists the user, encrypts the access token, and creates an application session.

## Session Cookies

Auth tokens are generated with `secrets.token_urlsafe(32)`. The database stores only a SHA-256 token hash. The default lifetime is 30 days and is configurable with `CODESEEK_AUTH_SESSION_TTL_SECONDS`.

Cookies are HTTP-only. `CODESEEK_AUTH_SESSION_SECURE_COOKIE=1` enables the secure flag and changes SameSite from `Lax` to `None`.

## Stored Credentials

The database stores encrypted GitHub, LLM provider, and embedding provider secrets. `crypto_store.py` derives keys from:

1. A request-scoped master-key override.
2. `CODESEEK_APP_ENCRYPTION_KEY`.
3. `APP_ENCRYPTION_KEY`.
4. `CODESEEK_API_KEY`.

Changing the effective key prevents existing secrets from being decrypted.

## Browser Secret Submission

`GET /api/v1/crypto/submission-key` returns an RSA public key and key ID. The browser can submit secrets encrypted with RSA-OAEP and SHA-256. A configured private key is loaded from `CODESEEK_SUBMISSION_PRIVATE_KEY_PEM` or `CODESEEK_SUBMISSION_PRIVATE_KEY_PATH`; otherwise, a process-local key is generated.

Plaintext submission is accepted only when `CODESEEK_ALLOW_PLAINTEXT_SECRET_SUBMISSION` is enabled.

## Authorization

Session and thread records carry user ownership. API, graph, trace, chat, and indexing endpoints check that ownership before returning or modifying resources. A non-visible session is returned as not found.
