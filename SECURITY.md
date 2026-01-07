# 🔐 Security Configuration Guide

## Overview

This project uses a secure secrets management system to protect API keys and sensitive credentials. Credentials are never committed to version control.

## File Structure

```
.env                  # Template for configuration (git-tracked)
.env.example         # Example configuration (git-tracked)
.secrets             # ACTUAL CREDENTIALS (git-ignored) ⚠️
.secrets.example     # Template for .secrets (git-tracked)
.gitignore           # Includes .secrets, .secrets.local, *.secrets
```

## Quick Setup

### 1. Copy Secrets Template

```bash
# On first setup, copy the example to .secrets
cp .secrets.example .secrets
```

### 2. Add Your Credentials

Edit `.secrets` and replace placeholder values with your actual API keys:

```bash
# .secrets file
GOOGLE_API_KEY=your_actual_gemini_key_here
OPENAI_API_KEY=your_actual_openai_key_here
```

### 3. Never Commit .secrets

The `.secrets` file is automatically git-ignored:

```bash
# .gitignore
.secrets           # Main secrets file
.secrets.local     # Local overrides
*.secrets          # Any .secrets variant
```

## How It Works

### Loading Order

1. **First**: Loads `.secrets` file (if it exists)
   - Contains actual API keys
   - Git-ignored for security
   - Local machine only

2. **Second**: Loads `.env` file
   - Template/default configuration
   - Safe to commit
   - Non-sensitive settings only

### Using SecureConfig

The project provides a `SecureConfig` class to safely load credentials:

```python
from src.secure_config import get_config

config = get_config()

# Get Gemini API key
gemini_key = config.get_gemini_key()

# Get OpenAI API key (optional)
openai_key = config.get_openai_key()

# Get any setting
model = config.get_setting("GEMINI_MODEL")

# Validate credentials
results = config.validate_credentials()
if results["valid"]:
    print("All credentials configured!")
else:
    print(f"Issues: {results['issues']}")
```

## Security Best Practices

### ✅ DO

- ✅ Store all API keys in `.secrets` file
- ✅ Use `.secrets.example` as a template
- ✅ Keep `.secrets` local (never share)
- ✅ Verify `.secrets` is in `.gitignore`
- ✅ Use environment variables for CI/CD
- ✅ Rotate keys periodically
- ✅ Use strong, unique credentials
- ✅ Check `.gitignore` before committing

### ❌ DON'T

- ❌ Commit `.secrets` file to git
- ❌ Hardcode credentials in source code
- ❌ Share API keys via email or chat
- ❌ Use the same key across environments
- ❌ Log credentials or expose in error messages
- ❌ Store credentials in `.env` (template only)
- ❌ Push credentials to public repositories
- ❌ Use placeholder keys in production

## Environment Variables

### For Development

Create `.secrets` with your actual keys:

```bash
GOOGLE_API_KEY=AIzaSyARr55SMfoGCE-NqLnJgNmwvqrvAb9xIfM
OPENAI_API_KEY=sk-your-key-here
```

### For CI/CD (GitHub Actions, etc.)

Set secrets in your CI/CD platform:

```bash
# GitHub Settings > Secrets > Actions
GOOGLE_API_KEY = AIzaSyARr55SMfoGCE-NqLnJgNmwvqrvAb9xIfM
OPENAI_API_KEY = sk-your-key-here
```

Then reference in workflows:

```yaml
env:
  GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Validation

Check if credentials are properly configured:

```bash
python -m src.secure_config
```

Output:
```
[INFO] Credential Validation Results:
============================================================
[✓] All required credentials are configured
[✓] Configured providers:
  - gemini
  - openai
============================================================
```

## Accidental Exposure Recovery

If you accidentally commit an API key:

### 1. Immediately Invalidate the Key

- Revoke the compromised key in your API provider dashboard
- Generate a new key
- Update `.secrets` with the new key

### 2. Remove from Git History

```bash
# Option 1: Remove file from history (recommended)
git filter-branch --tree-filter 'rm -f .secrets' HEAD

# Option 2: Use BFG Repo-Cleaner
bfg --delete-files .secrets

# Option 3: Force push after cleanup
git push origin main --force-with-lease
```

### 3. Update Team

- Notify team members to pull latest changes
- Update any deployed instances with new keys

## Multiple Environments

### Local Development

```bash
# .secrets (git-ignored)
GOOGLE_API_KEY=dev_key_xyz
```

### Staging

Use staging API keys in CI/CD secrets:

```bash
GOOGLE_API_KEY=staging_key_abc
```

### Production

Use production API keys in production CI/CD secrets:

```bash
GOOGLE_API_KEY=prod_key_123
```

## Troubleshooting

### Issue: "GOOGLE_API_KEY not configured"

**Solution:** Create `.secrets` file with your key:

```bash
cp .secrets.example .secrets
# Edit .secrets with your actual API key
```

### Issue: ".secrets not found"

**Warning:** It's OK if `.secrets` doesn't exist yet.

**Solution:** Create it:

```bash
cp .secrets.example .secrets
```

### Issue: Key is being logged

**Check:** Never log sensitive values:

```python
# ❌ DON'T
print(f"API Key: {api_key}")

# ✅ DO
print(f"API Key: ***{api_key[-4:]}")
```

## Key Rotation

Rotate API keys periodically:

1. Generate new key in provider dashboard
2. Add to `.secrets`
3. Test with new key
4. Delete old key from provider
5. Update team (if shared)

## Files Reference

| File | Committed | Purpose |
|------|-----------|---------|
| `.env` | ✅ Yes | Config template |
| `.env.example` | ✅ Yes | Example config |
| `.secrets` | ❌ No | **Actual credentials** |
| `.secrets.example` | ✅ Yes | Template for .secrets |
| `.gitignore` | ✅ Yes | Git ignore rules |

## Additional Resources

- [OWASP: Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [GitHub: Managing secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Python dotenv: Best practices](https://github.com/theskumar/python-dotenv)

---

**Last Updated:** January 7, 2026  
**Security Level:** Production-Grade  
**Status:** ✅ Implemented and Verified
