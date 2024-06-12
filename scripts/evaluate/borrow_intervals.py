import torch
from avm.search.audio import AudioIndex

def 

if __name__ == '__main__':
    audio_index = AudioIndex(index_embeddings_dir='data/rutube/audio_index_embeddings')

    query_embedding = torch.load('data/rutube/audio_val_embeddings/ydcrodwtz3mstjq1vhbdflx6kyhj3y0p.pt')

    print("query_embedding", query_embedding.shape)


