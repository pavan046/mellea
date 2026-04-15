# pytest: skip_always
"""Local LoRA adapter registration for Mellea's HF backend.

This module exists because Mellea's built-in adapter catalog assumes adapters
are fetched from HuggingFace Hub. For local checkpoints (e.g. trained LoRAs
that haven't been published), we need to register them directly by path.

LocalIntrinsicAdapter subclasses IntrinsicAdapter to override the path
resolution methods, pointing them at a local directory instead of performing
a Hub download. Config is loaded from an io.yaml file co-located with the
adapter weights — the same convention Mellea uses for Hub-hosted intrinsics.

Expected adapter directory layout:
    <adapter_path>/
        io.yaml                  # intrinsic config (response_format, parameters, ...)
        adapter_config.json      # LoRA architecture config
        adapter_model.safetensors

io.yaml extension: system_prompt
---------------------------------
Mellea's built-in io.yaml schema does not include a system_prompt field.
We extend the schema with an optional `system_prompt` key. If present,
LocalIntrinsicAdapter pops it from the config dict before passing the dict
to Mellea's internal validation (which would reject unknown fields), and
stores it separately as self.system_prompt. FCPipeline reads this attribute
to prepend a system message to the context before calling the LoRA.

This is preferable to hardcoding system prompts in pipeline code because it
keeps prompt ownership with the adapter author.

NOTE: This is a workaround for a missing Mellea feature. Ideally, Mellea
would support local adapter paths natively without requiring catalog
manipulation, and would support system_prompt natively in io.yaml. If either
is added upstream, this file can be simplified or removed.
"""

from __future__ import annotations

import os

import yaml

from mellea.backends.adapters.adapter import Adapter, IntrinsicAdapter
from mellea.backends.adapters.catalog import (
    _INTRINSICS_CATALOG,
    _INTRINSICS_CATALOG_ENTRIES,
    AdapterType,
    IntriniscsCatalogEntry,
)
from mellea.backends.huggingface import LocalHFBackend


class LocalIntrinsicAdapter(IntrinsicAdapter):
    """IntrinsicAdapter backed by a local LoRA checkpoint directory.

    Reads config from io.yaml co-located with the adapter weights.
    Registers the intrinsic in Mellea's global catalog on first use so the
    HF backend can resolve it by name. Subsequent instantiations with the
    same name reuse the existing catalog entry.

    Extends io.yaml with an optional `system_prompt` field not supported by
    Mellea's built-in schema. The field is popped from the config dict before
    Mellea's internal validation sees it and stored as self.system_prompt.
    FCPipeline reads self.system_prompt to prepend a system message to the
    context before calling the LoRA.

    Args:
        intrinsic_name: Logical name used to reference this adapter (e.g. "fc_router").
        adapter_path: Absolute path to the local LoRA checkpoint directory.
            Must contain an io.yaml file.
        adapter_type: LoRA by default.

    Raises:
        FileNotFoundError: if io.yaml is not found in adapter_path.
        ValueError: if io.yaml cannot be parsed as a YAML dict.
    """

    #: System prompt to prepend as a system message before calling this LoRA.
    #: None if not specified in io.yaml.
    system_prompt: str | None

    def __init__(
        self,
        intrinsic_name: str,
        adapter_path: str,
        adapter_type: AdapterType = AdapterType.LORA,
    ) -> None:
        io_yaml = os.path.join(adapter_path, "io.yaml")
        if not os.path.isfile(io_yaml):
            raise FileNotFoundError(
                f"No io.yaml found in adapter directory: {adapter_path!r}. "
                "Each adapter must have an io.yaml co-located with its weights."
            )

        with open(io_yaml, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        if not isinstance(raw_config, dict):
            raise ValueError(
                f"io.yaml in {adapter_path!r} did not parse to a dict. "
                f"Got {type(raw_config).__name__}."
            )

        # Pop system_prompt before Mellea validation sees the dict.
        # Mellea's make_config_dict raises ValueError on unknown top-level fields,
        # and system_prompt is our extension, not part of Mellea's schema.
        self.system_prompt = raw_config.pop("system_prompt", None)

        if intrinsic_name not in _INTRINSICS_CATALOG:
            entry = IntriniscsCatalogEntry(
                name=intrinsic_name, repo_id="local", adapter_types=(adapter_type,)
            )
            _INTRINSICS_CATALOG_ENTRIES.append(entry)
            _INTRINSICS_CATALOG[intrinsic_name] = entry

        Adapter.__init__(self, intrinsic_name, adapter_type)
        self.intrinsic_name = intrinsic_name
        self.intrinsic_metadata = _INTRINSICS_CATALOG[intrinsic_name]
        self.base_model_name = None
        self._adapter_path = adapter_path
        self.config = raw_config

    def get_local_hf_path(self, base_model_name: str) -> str:
        """Return the local checkpoint path directly."""
        return self._adapter_path

    def download_and_get_path(self, base_model_name: str) -> str:
        """Return the local checkpoint path (no download needed)."""
        return self._adapter_path


def ensure_adapter(
    name: str,
    path: str,
    backend: LocalHFBackend,
    adapter_type: AdapterType = AdapterType.LORA,
) -> None:
    """Register a LoRA adapter with the backend if not already registered.

    Reads adapter config from io.yaml co-located with the weights.

    Args:
        name: Logical adapter name (e.g. "fc_router").
        path: Local checkpoint directory path containing io.yaml.
        backend: The LocalHFBackend instance to register with.
        adapter_type: Adapter type, LoRA by default.
    """
    # TODO: replace print statements with proper logging (e.g. FancyLogger)
    qualified = f"{name}_{adapter_type.value}"
    if qualified not in backend._added_adapters:
        print(f"  [adapter] Loading {name} from {path}")
        backend.add_adapter(LocalIntrinsicAdapter(name, path, adapter_type))
        print(f"  [adapter] {name} loaded successfully")
    else:
        print(f"  [adapter] {name} already loaded, skipping")
