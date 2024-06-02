import torch
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")

class SequenceFingerPrinter():
    def __init__(self, config):
        pass

    def fingerprint(self, audio_or_video) -> torch.Tensor:
        pass

    # def 
