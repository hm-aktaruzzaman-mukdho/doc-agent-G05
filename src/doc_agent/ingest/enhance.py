"""Stage 1 — optional generative denoise or super-resolution enhancement."""

from __future__ import annotations

from ..contracts import Page


class Enhancer:
    """Safe enhancement seam; the current A2 baseline deliberately disables it."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["enhance"]
        self.trained_pages = 0

    def train(self, pages: list[Page]) -> None:
        """Record the training population for a future optional enhancer."""
        self.trained_pages = len(pages)

    def apply(self, pages: list[Page]) -> list[Page]:
        """Return pages unchanged when the configured enhancement type is ``none``."""
        enhancement_type = str(self.cfg.get("type", "none")).casefold()
        if enhancement_type not in {"none", "disabled", "identity"}:
            raise ValueError(
                "No trained generative enhancer is configured; set enhance.type to 'none' "
                "or provide the optional Stage 1 model before enabling it"
            )
        return pages


def run(pages: list[Page], cfg: dict) -> list[Page]:
    if not cfg["enhance"]["enabled"]:
        return pages
    return Enhancer(cfg).apply(pages)
