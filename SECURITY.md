# Security Policy

## Supported Versions

We actively maintain and patch the following versions:

| Version | Supported |
|---------|-----------|
| 2.x (latest) | ✅ Yes |
| 1.x | ❌ No — please upgrade |

---

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

If you discover a security vulnerability in Chakranetra, please report it responsibly by:

1. **Email**: Open a [GitHub Security Advisory](https://github.com/VijayabaskarR-06/chakranetra/security/advisories/new) (private, encrypted).
2. Include as much detail as possible:
   - A description of the vulnerability
   - Steps to reproduce
   - Potential impact (data exposure, privilege escalation, etc.)
   - Your suggested fix (optional but appreciated)

We will:
- Acknowledge receipt within **48 hours**
- Provide an estimated resolution timeline within **7 days**
- Credit you in the release notes (unless you prefer to remain anonymous)

---

## Scope

The following are **in scope** for security reports:

- Authentication bypass in `server/app.py`
- Arbitrary file upload / path traversal via `POST /api/scan/image`
- SQL injection or data leakage via the SQLite ticket registry
- Cross-origin data exposure on the dashboard
- Dependency vulnerabilities with a known CVE and a working exploit path

The following are **out of scope**:

- Vulnerabilities in dependencies without a direct exploit path
- Issues in the demo dataset (`data/samples/`) — it is intentionally public
- Rate limiting or DoS on a locally-hosted development instance
- Theoretical vulnerabilities without proof of concept

---

## Security Best Practices for Deployers

If you are deploying Chakranetra in a production municipal environment:

- **Run behind a reverse proxy** (nginx / Caddy) with TLS terminated at the proxy.
- **Set `ROADLENS_DB_PATH`** to a directory outside the web root with restricted filesystem permissions.
- **Restrict `POST /api/scan/image`** to authenticated vehicles/devices — do not expose it publicly without an auth layer.
- **Pin dependencies** using `pip-compile` or `poetry.lock` to prevent supply-chain attacks.
- **Rotate secrets** regularly and do not commit `.env` files to version control.

---

## Disclosure Timeline

We follow a **90-day coordinated disclosure** policy. If a fix cannot be shipped within 90 days, we will communicate publicly with mitigation guidance, even if a full patch is not yet ready.
