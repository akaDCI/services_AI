from .._base import BaseRestorationProvider, BaseConfig


class DefaultConfig(BaseConfig):
    ...


class DiffusionRestorationProvider(BaseRestorationProvider):
    def __init__(self, config=DefaultConfig):
        super().__init__(config)

    def initialize(self, *args, **kwargs):
        return super().initialize(*args, **kwargs)

    def infer(self, images, masks, server):
        raise NotImplementedError(
            "Diffusion restoration is not be implemented")
