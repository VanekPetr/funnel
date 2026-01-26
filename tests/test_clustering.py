"""Testing the clustering module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

from ifunnel.models.clustering import cluster, fancy_dendrogram, pick_cluster


def test_fancy_dendrogram_basic():
    """Test basic functionality of the fancy_dendrogram function."""
    data = [[1, 2], [2, 3], [6, 7], [8, 9]]
    z = linkage(data, method="complete")
    result = fancy_dendrogram(z, no_plot=True)
    assert isinstance(result, dict)
    assert "leaves" in result
    assert "dcoord" in result


def test_fancy_dendrogram_with_max_d():
    """Test fancy_dendrogram with max_d parameter."""
    data = [[1, 2], [2, 3], [6, 7], [8, 9]]
    z = linkage(data, method="complete")
    max_d = 5
    result = fancy_dendrogram(z, max_d=max_d, no_plot=True)
    assert "color_list" in result
    assert len(result["color_list"]) > 0


def test_fancy_dendrogram_plot_creation():
    """Ensure fancy_dendrogram creates a plot when no_plot=False."""
    data = [[1, 2], [2, 3], [6, 7], [8, 9]]
    z = linkage(data, method="complete")
    plt.figure()
    fancy_dendrogram(z, no_plot=False)
    assert plt.gcf().number == 1


def test_fancy_dendrogram_with_annotate_above():
    """Test fancy_dendrogram with annotate_above parameter."""
    data = [[1, 2], [2, 3], [6, 7], [8, 9]]
    z = linkage(data, method="complete")
    result = fancy_dendrogram(z, annotate_above=1, max_d=5, no_plot=False)
    assert isinstance(result, dict)
    plt.close()


def test_cluster_basic():
    """Test basic clustering functionality."""
    # Create sample return data
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(100, 5), columns=["A", "B", "C", "D", "E"])
    result = cluster(data, n_clusters=2, dendrogram=False)
    assert isinstance(result, pd.DataFrame)
    assert "Complete_Corr" in result.columns
    assert "Cluster" in result.columns
    assert len(result) == 5


def test_cluster_with_dendrogram():
    """Test clustering with dendrogram visualization."""
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(50, 4), columns=["A", "B", "C", "D"])
    plt.figure()
    result = cluster(data, n_clusters=2, dendrogram=True)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    plt.close()


def test_pick_cluster_basic():
    """Test basic pick_cluster functionality."""
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(100, 6), columns=["A", "B", "C", "D", "E", "F"])

    # Create cluster assignments
    ml = pd.DataFrame(
        {"Complete_Corr": [1, 1, 2, 2, 3, 3], "Cluster": ["Cluster 1"] * 2 + ["Cluster 2"] * 2 + ["Cluster 3"] * 2},
        index=["A", "B", "C", "D", "E", "F"],
    )

    # Create statistics
    stat = pd.DataFrame(
        {
            "Sharpe Ratio": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
        index=["A", "B", "C", "D", "E", "F"],
    )

    ids, result = pick_cluster(data, stat, ml, n_assets=1)
    assert isinstance(ids, list)
    assert len(ids) == 3  # One from each cluster
    assert isinstance(result, pd.DataFrame)


def test_pick_cluster_with_small_cluster():
    """Test pick_cluster when a cluster has fewer assets than requested."""
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(100, 4), columns=["A", "B", "C", "D"])

    # Create cluster assignments with one small cluster
    ml = pd.DataFrame(
        {"Complete_Corr": [1, 1, 1, 2], "Cluster": ["Cluster 1"] * 3 + ["Cluster 2"]},
        index=["A", "B", "C", "D"],
    )

    # Create statistics
    stat = pd.DataFrame(
        {"Sharpe Ratio": [0.5, 0.6, 0.7, 0.8]},
        index=["A", "B", "C", "D"],
    )

    ids, _result = pick_cluster(data, stat, ml, n_assets=2)
    assert len(ids) == 3  # 2 from Cluster 1, 1 from Cluster 2
