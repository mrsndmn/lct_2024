
from sklearn.decomposition import PCA
import torch
import os
import pickle

class PCAConfig:
    index_embeddings_path = 'data/music_caps/audio_embeddings'
    query_embeddings_path = 'data/music_caps/augmented_embeddings'
    n_components = 128

if __name__ == '__main__':

    config = PCAConfig()
    index_embeddings_path = config.index_embeddings_path
    query_embeddings_path = config.query_embeddings_path
    n_components = config.n_components

    embeddings_files = sorted(os.listdir(index_embeddings_path))
    augmented_embeddings_files = sorted(os.listdir(query_embeddings_path))

    all_embeddings = []
    for file_name in embeddings_files:
        embedding = torch.load(os.path.join(index_embeddings_path, file_name))
        all_embeddings.append(embedding)

    all_augmented_embeddings = []
    for file_name in augmented_embeddings_files:
        embedding = torch.load(os.path.join(query_embeddings_path, file_name))
        all_augmented_embeddings.append(embedding)

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_augmented_embeddings = torch.cat(all_augmented_embeddings, dim=0).numpy()

    print("all_embeddings", all_embeddings.shape)
    print("all_augmented_embeddings", all_augmented_embeddings.shape)

    pca = PCA(n_components=n_components)

    pca_embeddings = pca.fit_transform(all_embeddings)
    pca_augmented_embeddings = pca.transform(all_augmented_embeddings)

    print("pca_embeddings", pca_embeddings.shape)
    print("pca_augmented_embeddings", pca_augmented_embeddings.shape)

    with open(f'data/models/pca_{n_components}.pickle', 'wb') as f:
        pickle.dump(pca, f)

    base_pca_embeddings_path = f'data/music_caps/audio_embeddings_pca_{n_components}'
    os.makedirs(base_pca_embeddings_path, exist_ok=True)
    with open(f'{base_pca_embeddings_path}/audio_embeddings.pickle', 'wb') as f:
        pickle.dump(pca_embeddings, f)

    with open(f'{base_pca_embeddings_path}/augmented_audio_embeddings.pickle', 'wb') as f:
        pickle.dump(pca_augmented_embeddings, f)

    with open(f'{base_pca_embeddings_path}/audio_embeddings_files.pickle', 'wb') as f:
        pickle.dump(embeddings_files, f)

    with open(f'{base_pca_embeddings_path}/augmented_audio_embeddings_files.pickle', 'wb') as f:
        pickle.dump(augmented_embeddings_files, f)

