import pytest
from pydantic import ValidationError

from lynx_agent.config.schema import ProviderConfig


def test_provider_config_accepts_valid_values() -> None:
    config = ProviderConfig(
        api_key="test-secret",
        base_url="https://api.deepseek.com",
    )

    assert config.base_url.unicode_string() == "https://api.deepseek.com/"
    assert config.timeout_seconds == 60.0


def test_provider_config_masks_api_key() -> None:
    config = ProviderConfig(
        api_key="test-secret",
        base_url="https://api.deepseek.com",
    )

    assert str(config.api_key) == "**********"
    assert "test-secret" not in repr(config)


def test_provider_config_rejects_invalid_url() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig(
            api_key="test-secret",
            base_url="不是URL",
        )


def test_provider_config_rejects_invalid_timeout() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig(
            api_key="test-secret",
            base_url="https://api.deepseek.com",
            timeout_seconds=0,
        )