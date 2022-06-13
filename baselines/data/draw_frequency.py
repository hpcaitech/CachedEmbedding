import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

from ..dlrm_main import NUM_EMBEDDINGS_PER_FEATURE


def prepare_statistics(data):
    bins = sorted(np.log(np.bincount(data) + 1), reverse=True)    # plus 1 to avoid log(0)
    return bins


def draw_feature_frequency(sparse_data):
    num_feature = sparse_data.shape[1]

    fig, ax = plt.subplots(np.ceil(num_feature / 2).astype(int), 2, figsize=(16, 40))
    for col in range(num_feature):
        this_row, this_col = col // 2, col % 2
        this_ax = ax[this_row, this_col]
        this_feature = sparse_data[:, col]

        bins = prepare_statistics(this_feature)
        this_ax.scatter(np.arange(len(bins)), bins, s=0.5, marker='+')
        this_ax.set_xlabel("feature ids sorted by frequency")
        this_ax.set_ylabel('log(frequency) of sparse feature ids')

    fig.savefig('separate.svg', format="svg", bbox_inches="tight")
    plt.close(fig)


def draw_overall_frequency(sparse_data):
    num_embeddings_per_feature = list(map(int, NUM_EMBEDDINGS_PER_FEATURE.split(",")))
    offsets = np.array([0, *np.cumsum(num_embeddings_per_feature)[:-1]]).reshape(1, -1)

    global_data = (sparse_data + offsets).reshape(-1)
    bins = prepare_statistics(global_data)[:10000]

    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(bins)), bins, s=0.5, marker='+')
    ax.set_xlabel("Top 10k sparse feature ids sorted by frequency")
    ax.set_ylabel("log(frequency)")
    ax.set_title("Statistics of sparse feature id")

    fig.savefig('global.svg', format='svg', bbox_inches="tight")
    plt.close(fig)


def main(path):
    # Load data
    sparse_data = []
    for each in os.listdir(path):
        if 'sparse' in each:
            sparse_data.append(np.load(os.path.join(path, each)))
    sparse_data = np.vstack(sparse_data)
    num_embeddings_per_feature = np.array([list(map(int, NUM_EMBEDDINGS_PER_FEATURE.split(",")))])
    sparse_data %= num_embeddings_per_feature
    print(f"sparse data shape: {sparse_data.shape}")

    # print([min(sparse_data[c]) for c in range(sparse_data.shape[1])])
    # TODO: draw_feature_frequency(sparse_data)
    draw_overall_frequency(sparse_data)


if __name__ == '__main__':
    main('criteo_kaggle')
