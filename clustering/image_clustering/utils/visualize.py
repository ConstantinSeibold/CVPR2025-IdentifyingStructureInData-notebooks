import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import math
from PIL import Image
import numpy as np

import math
from PIL import Image

from scipy.stats import mode

def create_image_grid(image_data, images_per_cluster, rows=None, cols=None, resize_to=(224, 224)):
    """
    Creates a grid of resized images based on the given image data.

    Parameters:
    - image_data (list): List of image file paths or PIL Image objects.
    - images_per_cluster (int): Number of images to display in the grid.
    - rows (int, optional): Number of rows in the grid. Default is ceil(sqrt(images_per_cluster)).
    - cols (int, optional): Number of columns in the grid. Default is ceil(sqrt(images_per_cluster)).
    - resize_to (tuple, optional): The size to which each image will be resized. Default is (224, 224).

    Returns:
    - PIL Image: The resulting grid image.
    """

    if rows is None or cols is None:
        grid_size = math.ceil(math.sqrt(images_per_cluster))
        rows = cols = grid_size

    # Resize all images to the desired size
    resized_images = [img.resize(resize_to) if isinstance(img, Image.Image) else Image.open(img).resize(resize_to) for img in image_data]

    # Calculate the grid size (row x col)
    grid_width = cols * resize_to[0]  # width of each image is resize_to[0]
    grid_height = rows * resize_to[1]  # height of each image is resize_to[1]

    # Create a new blank image with a white background
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    # Paste the resized images into the grid
    for i in range(min(images_per_cluster, len(resized_images))):
        row = i // cols
        col = i % cols
        img = resized_images[i]

        # Calculate the position to paste the image
        x_offset = col * resize_to[0]
        y_offset = row * resize_to[1]

        grid_image.paste(img, (x_offset, y_offset))

    return grid_image

    
def plot_cluster_images(dataset_samples, cluster_labels, cluster_names = None, clusters_per_row=3, images_per_cluster=4):
    """
    Visualizes images grouped by their cluster labels.
    
    Args:
        image_data: List or array of image data (could be numpy arrays or image file paths).
        cluster_labels: List or array of cluster labels corresponding to each image.
        clusters_per_row: The number of clusters to display in each row (integer).
        images_per_cluster: The number of images to display per cluster (integer).
    """
    image_data = [i[0] for i in dataset_samples]
    
    unique_clusters = np.unique(cluster_labels)
    
    # Initialize grid size based on clusters per row
    rows = int(np.ceil(len(unique_clusters) / clusters_per_row))
    
    # Create a new figure for the plot
    fig, axes = plt.subplots(rows, clusters_per_row, figsize=(clusters_per_row * 3, rows * 3))
    axes = axes.flatten()  # Flatten to easily iterate over axes

    # Plot images for each cluster
    for i, cluster in enumerate(unique_clusters):
        ax = axes[i]
        
        # Find images in the current cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        
        # Select the first few images from this cluster
        merged_image = create_image_grid(np.array(image_data)[cluster_indices[:images_per_cluster]], images_per_cluster)
        # for j, idx in enumerate(cluster_indices[:images_per_cluster]):
        ax.imshow(merged_image)  # Show the image
        # ax.imshow(Image.open(image_data[idx]))  # Show the image
        ax.axis('off')  # Hide the axis

        if cluster_names is None:
            ax.set_title(f"Cluster {cluster}")  # Title with the cluster number
        else:
            ax.set_title(f"{cluster_names[i]}")  # Title with the cluster number
        
        # If there are no images left in this cluster, make sure the axis is turned off
        if len(cluster_indices) == 0:
            ax.axis('off')
    
    # Hide any unused axes
    for i in range(len(unique_clusters), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

def apply_pca(features: np.ndarray,
              n_components: int = 50,
              whiten: bool = True,
              random_state: int = 42,
              verbose=False
             ) -> tuple[np.ndarray, PCA]:
    """
    Perform PCA on input data matrix X.

    Parameters
    ----------
    features : array-like of shape (n_samples, n_features)
        Input data, where each row is a sample and each column is a feature.
    n_components : int, default=50
        Number of principal components to keep.
    whiten : bool, default=True
        When True, the components_ vectors are multiplied by
        n_samples**0.5 and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.
    random_state : int, default=42
        Seed for the random number generator (only used when svd_solver='randomized').

    Returns
    -------
    features_reduced : ndarray of shape (n_samples, n_components)
        The projected data in the principal component space.
    pca : PCA object
        Fitted PCA instance; use `pca.explained_variance_ratio_` to inspect variance explained.
    """
    pca = PCA(n_components=n_components,
              whiten=whiten,
              random_state=random_state)
    features_reduced = pca.fit_transform(features)
    return features_reduced

def apply_tsne(features, n_components=2, perplexity=30, random_state=42, verbose=False):
    """
    Applies t-SNE to reduce the feature dimensions to 2D (or specified).
    
    Args:
        features (np.ndarray): Feature matrix of shape (n_samples, n_features).
        n_components (int): Target number of dimensions for t-SNE.
        perplexity (float): t-SNE perplexity parameter.
        random_state (int): Random state for reproducibility.
    
    Returns:
        np.ndarray: Transformed 2D (or n_components) feature matrix.
    """
    if verbose:
        print('Applying Dimensionality reduction')
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    reduced_features = tsne.fit_transform(features)
    return reduced_features

def plot_tsne(reduced_features, labels, title="t-SNE Visualization"):
    """
    Plots the 2D t-SNE reduced features with coloring by labels.
    
    Args:
        reduced_features (np.ndarray): 2D t-SNE output (n_samples, 2).
        labels (np.ndarray or list): Labels for each sample.
        title (str): Title of the plot.
    """
    print('Visualize Embeddings')
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_multiple_tsne(reduced_features_list, labels_list, titles=None, num_columns=5):
    """
    Plots multiple t-SNE visualizations with a specified number of columns.
    
    Args:
        reduced_features_list (list of np.ndarray): List containing 2D t-SNE outputs for each dataset (n_samples, 2).
        labels_list (list of np.ndarray or list): List of labels for each sample in each dataset.
        titles (list of str, optional): Titles for each plot. If None, default title is used.
        num_columns (int, optional): Number of columns for the subplot grid. Defaults to 2.
    """
    
    num_plots = len(reduced_features_list)
    num_rows = (num_plots + num_columns - 1) // num_columns
    
    # Set up the figure and axes
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(8 * num_columns, 6 * num_rows))
    axes = axes.flatten() if num_plots > 1 else [axes]
    
    if titles is None:
        titles = [f"t-SNE Visualization {i+1}" for i in range(num_plots)]
    
    for idx, (reduced_features, labels) in enumerate(zip(reduced_features_list, labels_list)):
        
        ax = axes[idx]
        
        scatter = ax.scatter(
            reduced_features[:, 0], 
            reduced_features[:, 1], 
            c=labels, 
            cmap='tab10', 
            alpha=0.7
        )
        ax.set_title(titles[idx])
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        ax.grid(True)
        
    # Hide any unused subplots
    for ax in axes[num_plots:]:
        ax.axis('off')
    
    # Create a common legend for all subplots
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.7)
    fig.legend(handles, titles[0].split(' ')[-1] + " Classes", loc='upper right')
    
    plt.tight_layout()
    plt.show()


def plot_cluster_label_matches(reduced_features, true_labels, cluster_preds, title="Cluster vs Label Matching"):
    """
    Visualizes how well predicted clusters match true labels using t-SNE 2D output.
    
    Args:
        reduced_features (np.ndarray): 2D t-SNE output (n_samples, 2).
        true_labels (np.ndarray): True class labels.
        cluster_preds (np.ndarray): Cluster assignments (e.g., from k-means).
        title (str): Title of the plot.
    """
    # Step 1: Match clusters to most common true label
    matched_labels = np.zeros_like(cluster_preds)
    unique_clusters = np.unique(cluster_preds)
    
    for cluster in unique_clusters:
        mask = cluster_preds == cluster
        most_common = mode(true_labels[mask], keepdims=True).mode[0]
        matched_labels[mask] = most_common
    
    # Step 2: Create boolean mask for correct predictions
    correct = matched_labels == true_labels
    
    # Step 3: Plotting
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=matched_labels,
        cmap='tab10',
        edgecolors=np.where(correct, 'black', 'red'),
        linewidths=0.6,
        alpha=0.8
    )
    
    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.legend(*scatter.legend_elements(), title="Matched Label")
    plt.tight_layout()
    plt.show()
    

def plot_label_distributions(label_dict):
    """
    Plots the distribution of cluster sizes for each clustering method.
    
    Parameters:
        label_dict (dict): A dictionary where keys are method names (str) and values are label arrays (np.ndarray).
    """
    num_methods = len(label_dict)
    ncols = 5
    nrows = int(np.ceil(num_methods / ncols))

    plt.figure(figsize=(5 * ncols, 4 * nrows))
    
    for i, (method, labels) in enumerate(label_dict.items(), 1):
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        plt.subplot(nrows, ncols, i)
        sns.barplot(x=list(cluster_sizes.keys()), y=list(cluster_sizes.values()))
        plt.title(f'{method} Cluster Sizes')
        plt.xlabel('Cluster Label')
        plt.ylabel('Number of Points')
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.suptitle('Cluster Size Distributions by Method', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_multiple_tsne_variants(reduced_features, label_sets, titles=None, n_cols=2):
    """
    Plots multiple sets of labels on the same t-SNE reduced features.
    
    Args:
        reduced_features (np.ndarray): 2D t-SNE output (n_samples, 2).
        label_sets (list of np.ndarray): List of label arrays to plot.
        titles (list of str): Titles for each subplot.
        n_cols (int): Number of columns in subplot layout.
    """
    n_variants = len(label_sets)
    n_rows = int(np.ceil(n_variants / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    
    for idx, labels in enumerate(label_sets):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1],
                             c=labels, cmap='tab10', alpha=0.7)
        if titles:
            ax.set_title(titles[idx])
        else:
            ax.set_title(f"Label Set {idx + 1}")
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        ax.grid(True)
    
    # Hide any unused subplots
    for idx in range(n_variants, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.show()


def plot_cluster_label_matches_variants(reduced_features, true_labels, cluster_sets, titles=None, n_cols=2):
    """
    Vectorized version: plots multiple cluster sets with matched‐label overlay,
    using only two calls to `ax.scatter` per subplot instead of looping over points.
    """
    n_variants = len(cluster_sets)
    n_rows = int(np.ceil(n_variants / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    # Precompute a colormap
    cmap = plt.cm.tab10

    for idx, clusters in enumerate(cluster_sets):
        ax = axes[idx // n_cols, idx % n_cols]

        # Compute the mode label for each cluster (vectorized)
        # `mode` returns an array of modes for each unique cluster
        uniq, inv = np.unique(clusters, return_inverse=True)
        modes = np.array([np.bincount(true_labels[clusters==c]).argmax()
                          for c in uniq])
        matched = modes[inv]

        # Which points are correctly labeled?
        correct = (matched == true_labels)

        # scatter dots colored by correctness
        ax.scatter(
            reduced_features[:, 0], reduced_features[:, 1],
            s=40,
            c=np.where(correct, 'blue', 'red'),
            edgecolor='none'
        )

        # titles, labels, grid
        title = titles[idx] if titles else f"Clustering {idx+1}"
        ax.set(title=title, xlabel="t-SNE Dim 1", ylabel="t-SNE Dim 2")
        ax.grid(True)

    # turn off any extra axes
    for idx in range(n_variants, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis('off')

    fig.tight_layout()
    plt.show()

def build_results_dataframe(results, times, memory):
    """
    Builds a pandas DataFrame from clustering evaluation results.
    
    Args:
        results (dict): Mapping from method name to dict of metrics (ARI, NMI, FMI, Silhouette).
        times (dict): Mapping from method name to runtime in seconds.
        memory (dict): Mapping from method name to RAM usage in MB.
        
    Returns:
        pd.DataFrame: DataFrame with columns: Method, Time, RAM, ARI, NMI, FMI, Silhouette.
    """
    data = []
    for method, scores in results.items():
        row = {
            'Method': method,
            'Time (s)': times.get(method, None),
            'RAM (MB)': memory.get(method, None),
            'ARI': scores.get('ARI', None),
            'NMI': scores.get('NMI', None),
            'FMI': scores.get('FMI', None),
            'Silhouette': scores.get('Silhouette', None),
        }
        data.append(row)
    df = pd.DataFrame(data).set_index('Method')
    return df

def plot_performance_metrics(df):
    """
    Plots ARI, NMI, FMI, and Silhouette scores as grouped bars.
    
    Args:
        df (pd.DataFrame): DataFrame indexed by Method.
    """
    metrics = ['ARI', 'NMI', 'FMI', 'Silhouette']
    ax = df[metrics].plot(kind='bar', figsize=(8, 6))
    ax.set_ylabel('Score')
    ax.set_title('Clustering Performance Metrics')
    ax.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_resource_metrics(df):
    """
    Plots Time and RAM usage as grouped bars.
    
    Args:
        df (pd.DataFrame): DataFrame indexed by Method.
    """
    resources = ['Time (s)', 'RAM (MB)']
    ax = df[resources].plot(kind='bar', figsize=(8, 6))
    ax.set_ylabel('Resource Usage')
    ax.set_title('Clustering Resource Consumption')
    ax.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_efficiency_vs_performance(df, x_metric='Time (s)', y_metric='NMI', log_x=False):
    """
    Plots a 2D scatter plot of resource usage vs. clustering performance.
    
    Args:
        df (pd.DataFrame): DataFrame with clustering evaluation results.
        x_metric (str): Metric to use on x-axis (e.g., 'Time (s)' or 'RAM (MB)').
        y_metric (str): Metric to use on y-axis (e.g., 'NMI', 'ARI', etc.).
        log_x (bool): Whether to use a logarithmic scale on the x-axis.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_metric, y=y_metric, hue=df.index, s=100, palette='tab10')
    
    for method in df.index:
        plt.text(df.loc[method, x_metric], df.loc[method, y_metric], method, fontsize=9,
                 ha='right' if df.loc[method, x_metric] > df[x_metric].mean() else 'left',
                 va='bottom')
    
    if log_x:
        plt.xscale('log')
    
    plt.title(f'{y_metric} vs {x_metric} {"(log scale)" if log_x else ""}')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
