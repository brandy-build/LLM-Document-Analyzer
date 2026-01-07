"""Secure configuration loader for API keys and secrets"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class SecureConfig:
    """
    Securely loads credentials from .secrets file
    Follows security best practices:
    - Never hardcodes credentials
    - Never exposes secrets in logs
    - Uses .secrets file (git-ignored)
    - Falls back to .env for non-sensitive config
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SecureConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._secrets = {}
        self._load_secrets()
        self._initialized = True

    def _load_secrets(self):
        """Load secrets from .secrets file first, then .env"""
        project_root = Path(__file__).parent.parent

        # Try to load .secrets file first
        secrets_file = project_root / ".secrets"
        if secrets_file.exists():
            logger.info("Loading credentials from .secrets file")
            load_dotenv(dotenv_path=secrets_file, override=True)
        else:
            logger.warning(
                f".secrets file not found at {secrets_file}. "
                "Copy .secrets.example to .secrets and add your credentials"
            )

        # Fall back to .env for non-sensitive config
        env_file = project_root / ".env"
        if env_file.exists():
            logger.info("Loading configuration from .env file")
            load_dotenv(dotenv_path=env_file, override=False)

    def get_gemini_key(self) -> str:
        """Get Gemini API key with security checks"""
        key = os.getenv("GOOGLE_API_KEY")
        if not key or key.startswith("your_"):
            raise ValueError(
                "GOOGLE_API_KEY not configured. "
                "Please copy .secrets.example to .secrets and add your key"
            )
        return key

    def get_openai_key(self) -> Optional[str]:
        """Get OpenAI API key (optional)"""
        key = os.getenv("OPENAI_API_KEY")
        if key and key.startswith("your_"):
            return None
        return key

    def get_gemini_model(self) -> str:
        """Get Gemini model name"""
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def get_openai_model(self) -> str:
        """Get OpenAI model name"""
        return os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get any configuration setting

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        return os.getenv(key, default)

    def validate_credentials(self) -> dict:
        """
        Validate that required credentials are configured

        Returns:
            Dictionary with validation results
        """
        results = {"valid": True, "issues": [], "configured": {}}

        # Check Gemini API key
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_key or gemini_key.startswith("your_"):
            results["valid"] = False
            results["issues"].append("GOOGLE_API_KEY not configured")
        else:
            results["configured"]["gemini"] = True

        # Check OpenAI API key (optional)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and not openai_key.startswith("your_"):
            results["configured"]["openai"] = True
        elif openai_key:
            results["issues"].append("OPENAI_API_KEY placeholder found - remove or replace")

        return results

    @staticmethod
    def get_secrets_path() -> Path:
        """Get the path to .secrets file"""
        return Path(__file__).parent.parent / ".secrets"

    @staticmethod
    def create_secrets_from_example() -> bool:
        """
        Helper to create .secrets from .secrets.example

        Returns:
            True if successful, False otherwise
        """
        project_root = Path(__file__).parent.parent
        example_file = project_root / ".secrets.example"
        secrets_file = project_root / ".secrets"

        if secrets_file.exists():
            logger.warning(".secrets file already exists")
            return False

        if not example_file.exists():
            logger.error(".secrets.example not found")
            return False

        try:
            secrets_file.write_text(example_file.read_text())
            logger.info(f"Created {secrets_file} from {example_file}")
            logger.warning("⚠️  Please edit .secrets and add your actual credentials")
            return True
        except Exception as e:
            logger.error(f"Failed to create .secrets: {e}")
            return False


# Global instance
config = SecureConfig()


def get_config() -> SecureConfig:
    """Get the global SecureConfig instance"""
    return config


if __name__ == "__main__":
    # Test the configuration
    logging.basicConfig(level=logging.INFO)
    cfg = SecureConfig()

    print("\n[INFO] Credential Validation Results:")
    print("=" * 60)

    results = cfg.validate_credentials()

    if results["valid"]:
        print("[OK] All required credentials are configured")
    else:
        print("[WARNING] Configuration issues found:")
        for issue in results["issues"]:
            print(f"  - {issue}")

    if results["configured"]:
        print("\n[OK] Configured providers:")
        for provider in results["configured"]:
            print(f"  - {provider}")

    print("=" * 60)
