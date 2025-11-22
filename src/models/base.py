from abc import ABC, abstractmethod


class BasePolicy(ABC):
    @abstractmethod
    def forward(self, batch):
        pass

    @abstractmethod
    def predict_action(self, state_seq, return_seq, timesteps):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def count_spikes(self):
        pass

    @abstractmethod
    def num_params(self):
        pass