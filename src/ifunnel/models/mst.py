"""Module for minimum spanning tree (MST) based asset selection.

This module provides functions to select a diversified subset of financial assets
using graph theory, specifically the minimum spanning tree algorithm. It identifies
assets that are less correlated with each other, which is useful for portfolio
diversification.
"""

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA


def minimum_spanning_tree(dataset: pd.DataFrame) -> tuple[list[str], pd.DataFrame, float, float]:
    """Select a diversified subset of assets using minimum spanning tree algorithm.

    This function constructs a graph where nodes are assets and edge weights are
    distances based on correlation. It then computes a minimum spanning tree and
    selects leaf nodes (degree 1) as the diversified subset of assets.

    Args:
        dataset: DataFrame containing financial time series data, typically returns,
                with assets as columns and time periods as rows

    Returns:
        Tuple containing:
            - List[str]: List of selected asset identifiers (leaf nodes)
            - pd.DataFrame: DataFrame containing only the selected assets' data
            - float: Average correlation among the selected assets
            - float: Portfolio Diversification Index (PDI) for the selected assets

    Example:
        One asset is kept from each of two anti-correlated pairs:

        >>> import numpy as np
        >>> import pandas as pd
        >>> rng = np.random.default_rng(0)
        >>> base = rng.normal(size=(60, 1))
        >>> noise = 0.1 * rng.normal(size=(60, 4))
        >>> data = pd.DataFrame(np.hstack([base, base, -base, -base]) + noise, columns=list("ABCD"))
        >>> selected, subset, avg_corr, pdi = minimum_spanning_tree(data)
        >>> len(set(selected) & {"A", "B"}), len(set(selected) & {"C", "D"})
        (1, 1)
        >>> list(subset.columns) == selected
        True
    """
    logger.debug("💡 Running MST method")

    corr = dataset.corr(method="spearman")  # calculate the correlation
    distance_corr = (2 * (1 - corr)) ** 0.5  # calculate the distance
    mask = np.triu(np.ones_like(corr, dtype=bool))  # get only the upper half of the matrix
    distance_corr = distance_corr * mask

    # use the correlation matrix to create links
    links = distance_corr.stack().reset_index(level=1)
    links.columns = ["var2", "value"]
    links = links.reset_index()
    links = links.replace(0, np.nan)  # drop 0 values from the matrix
    links = links.dropna(how="any", axis=0)
    links.columns = ["var1", "var2", "value"]  # rename the columns
    links_filtered = links.loc[(links["var1"] != links["var2"])]  # filter out self-correlations

    # Create the graph
    created_graph = nx.Graph()
    for i in range(len(corr)):  # add nodes
        created_graph.add_node(corr.index[i])
    tuples = list(links_filtered.itertuples(index=False, name=None))  # add edges with weight
    created_graph.add_weighted_edges_from(tuples)

    # Create a MST from the full graph
    mst = nx.minimum_spanning_tree(created_graph)

    # Save the nodes with degree one
    degrees = [val for (node, val) in mst.degree()]
    df = pd.DataFrame(degrees, corr.index)
    df.columns = ["degree"]
    subset = df[df["degree"] == 1].index.tolist()

    # Create a new dataframe with only the assets from the subset
    subset_df = dataset.loc[:, dataset.columns.isin(subset)]

    # Calculate the average correlation of the subset
    corr_subset = subset_df.corr(method="spearman")
    corr_avg = corr_subset.mean().mean()

    # Calculate the PDI for the subset
    pca = PCA()
    pca.fit(corr_subset)
    value = 0
    for i in range(1, corr_subset.shape[1]):
        value = value + i * pca.explained_variance_ratio_[i - 1]
    pdi = 2 * value - 1

    return subset, subset_df, corr_avg, pdi
