"""
Dynamic OpenAI-compatible model handler.
Based on ToolSandbox's dynamic model creation approach.
"""
import os
from typing import Optional

from bfcl_eval.constants.model_config import ModelConfig
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from openai import OpenAI


def create_dynamic_handler_class(api_key=None, base_url=None, is_fc_model=False):
    """
    Create a dynamic handler class with pre-configured API settings.

    Args:
        api_key: API key for the model service
        base_url: Base URL for the API endpoint
        is_fc_model: Whether this is a function calling model

    Returns:
        A handler class that can be instantiated with standard (model_name, temperature) signature
    """
    class DynamicOpenAIHandler(OpenAICompletionsHandler):
        def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
            # Store dynamic configuration
            self._dynamic_api_key = api_key or os.getenv("API_KEY")
            self._dynamic_base_url = base_url or os.getenv("BASE_URL")
            self._is_fc_model_dynamic = is_fc_model

            # Temporarily set OPENAI_ environment variables for parent init
            old_api_key = os.environ.get("OPENAI_API_KEY")
            old_base_url = os.environ.get("OPENAI_BASE_URL")

            if self._dynamic_api_key:
                os.environ["OPENAI_API_KEY"] = self._dynamic_api_key
            if self._dynamic_base_url:
                os.environ["OPENAI_BASE_URL"] = self._dynamic_base_url

            try:
                # Call parent init
                super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
            finally:
                # Restore original environment variables
                if old_api_key is not None:
                    os.environ["OPENAI_API_KEY"] = old_api_key
                elif "OPENAI_API_KEY" in os.environ:
                    del os.environ["OPENAI_API_KEY"]

                if old_base_url is not None:
                    os.environ["OPENAI_BASE_URL"] = old_base_url
                elif "OPENAI_BASE_URL" in os.environ:
                    del os.environ["OPENAI_BASE_URL"]

            # Set the is_fc_model property
            self.is_fc_model = self._is_fc_model_dynamic

    return DynamicOpenAIHandler


def create_dynamic_model_config(
    model_name: str,
    display_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    is_fc_model: bool = False,
    underscore_to_dot: bool = False,
    org: str = "Custom",
    license: str = "Unknown"
):
    """
    Create a dynamic model configuration.

    Args:
        model_name: The model name to use for API calls (should NOT contain the -FC suffix)
        display_name: Display name for the leaderboard (defaults to model_name)
        api_key: API key for the model service
        base_url: Base URL for the API endpoint
        is_fc_model: Whether this is a function calling model
        underscore_to_dot: Whether to replace '.' with '_' in function names (typically True for FC models)
        org: Organization name
        license: License information

    Returns:
        A ModelConfig object for the dynamic model
    """
    # Create a dynamic handler class with pre-configured API settings
    handler_class = create_dynamic_handler_class(
        api_key=api_key,
        base_url=base_url,
        is_fc_model=is_fc_model
    )

    mode_label = "FC" if is_fc_model else "Prompt"
    return ModelConfig(
        model_name=model_name,
        display_name=display_name or f"{model_name} (Dynamic {mode_label})",
        url=base_url or "Custom API",
        org=org,
        license=license,
        model_handler=handler_class,
        input_price=None,  # Unknown for custom models
        output_price=None,  # Unknown for custom models
        is_fc_model=is_fc_model,
        underscore_to_dot=underscore_to_dot,
    )


FC_SUFFIX = "-FC"


def ensure_dynamic_models_registered(model_names: list[str]):
    """
    Register dynamic model configs for any model names not already in MODEL_CONFIG_MAPPING.

    Models ending with '-FC' are registered in function-calling mode:
    - The suffix is stripped to derive the actual API model name
    - is_fc_model=True and underscore_to_dot=True are set

    This function is shared by both the generation and evaluation pipelines
    to avoid duplicating the registration logic.
    """
    import bfcl_eval.constants.model_config
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

    extended = MODEL_CONFIG_MAPPING.copy()
    changed = False

    for registry_name in model_names:
        if registry_name in MODEL_CONFIG_MAPPING:
            continue

        api_key = os.getenv("API_KEY")
        base_url = os.getenv("BASE_URL")

        if not (api_key and base_url):
            raise ValueError(
                f"Unknown model_name '{registry_name}'.\n"
                "• For officially supported models, please refer to `SUPPORTED_MODELS.md`.\n"
                "• For running new models, please refer to `README.md` and `CONTRIBUTING.md`.\n"
                "• For dynamic models, set environment variables: API_KEY and BASE_URL"
            )

        is_fc = registry_name.endswith(FC_SUFFIX)
        api_model_name = registry_name[: -len(FC_SUFFIX)] if is_fc else registry_name

        dynamic_config = create_dynamic_model_config(
            model_name=api_model_name,
            api_key=api_key,
            base_url=base_url,
            is_fc_model=is_fc,
            underscore_to_dot=is_fc,
        )
        extended[registry_name] = dynamic_config
        mode = "FC" if is_fc else "Prompt"
        print(f"Created dynamic configuration for model: {registry_name} ({mode} mode, api_name={api_model_name})")
        changed = True

    if changed:
        bfcl_eval.constants.model_config.MODEL_CONFIG_MAPPING = extended