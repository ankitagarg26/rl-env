from abc import ABC, abstractmethod

class BaseAlgo(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def collect_experiences(self):
        pass
    
    @abstractmethod
    def update_parameters(self):
        pass
