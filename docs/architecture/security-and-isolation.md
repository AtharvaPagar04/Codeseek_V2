# Security And Isolation

## Request Authentication

The API supports an application bearer key and GitHub-backed user sessions. User-scoped endpoints resolve the HTTP-only auth cookie or accepted authorization path and reject resources owned by another user.

## Secret Storage

GitHub tokens, generation provider keys, and embedding provider keys are encrypted through `retrieval/stores/crypto_store.py`. The implementation derives separate SHA-256 encryption and MAC keys, uses a nonce-based XOR keystream, and authenticates each payload with HMAC-SHA-256.

The submission-key endpoint exposes an RSA public key. It loads configured key material when present or generates a process-local key otherwise. Clients can submit RSA-OAEP encrypted secrets. Plaintext secret submission is controlled by `CODESEEK_ALLOW_PLAINTEXT_SECRET_SUBMISSION`.

## Session Tokens

Application session tokens are random values. Only their SHA-256 hashes are persisted. Cookies are HTTP-only. They use `SameSite=Lax` by default and `SameSite=None` when `CODESEEK_AUTH_SESSION_SECURE_COOKIE` enables secure cookies.

## Repository Isolation

`retrieval/support/isolation.py` derives the expected collection from tenant and repository identity. Query requests validate the collection against the active repository root before retrieval.

Session, thread, graph, trace, credential, and repository endpoints check user visibility before returning data.

## Transport Controls

- CORS uses `CODESEEK_CORS_ORIGINS` or the legacy `CORS_ALLOWED_ORIGINS`.
- HTTPS enforcement is controlled by `CODESEEK_ENFORCE_HTTPS`.
- Forwarded protocol trust is controlled by `CODESEEK_TRUST_X_FORWARDED_PROTO`.
- Request rate limiting uses `CODESEEK_RATE_LIMIT_PER_MINUTE`.

## Required Production Settings

Use a stable high-entropy application encryption key, secure cookies, HTTPS enforcement, restricted CORS origins, non-default database credentials, and encrypted secret submission. Changing the encryption key makes existing encrypted credentials unreadable.
