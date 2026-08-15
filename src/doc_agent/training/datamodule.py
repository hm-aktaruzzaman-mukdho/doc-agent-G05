"""Training — Lightning datamodule"""

from __future__ import annotations

import lightning

from ..contracts import *  # noqa


class DocDataModule(lightning.LightningDataModule):
    def setup(self, stage: str | None = None) -> None:
        raise NotImplementedError("Training: datamodule.setup (split by document)")
