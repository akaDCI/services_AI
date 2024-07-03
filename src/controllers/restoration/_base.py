from abc import abstractmethod
from typing import List
import numpy as np
import numpy.typing as npt
from ._enum import InferenceServer


class BaseConfig:
    def show_configs(self):
        # Get all instance attributes
        instance_attributes = vars(self)

        # Get all class attributes (excluding built-in attributes)
        class_attributes = {k: v for k, v in vars(
            self.__class__).items() if not k.startswith('__') and not callable(v)}

        # Combine both dictionaries, giving priority to instance attributes
        combined_attributes = {**class_attributes, **instance_attributes}

        # Print table header
        print(f"| {'Name':<15} | {'Type':<15} | {'Value':<15} |")
        print(f"| {'-'*15} | {'-'*15} | {'-'*15} |")

        # Print each attribute in table format
        for name, value in combined_attributes.items():
            value_type = type(value).__name__
            print(f"| {name:<15} | {value_type:<15} | {str(value):<15} |")


class BaseRestorationProvider:
    def __init__(self, config: BaseConfig = BaseConfig()):
        """
        Initialize model, variables, configs
        """
        self.config = config

    @abstractmethod
    def initialize(self, *args, **kwargs):
        """
        Download model resources.
        """

    @abstractmethod
    def infer(
        self,
        images: List[npt.NDArray[np.uint8]],
        masks: List[npt.NDArray[np.uint8]],
        server: InferenceServer,
    ) -> List[npt.NDArray]:
        """
        Inference method of inpainting.
        """
