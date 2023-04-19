from sklearn.cluster import KMeans
import numpy as np

class ImageCluster:
    """
    The ImageCluster class performs clustering of images based on their feature vectors.

    Parameters
    ----------
    max_clusters : int
        The maximum number of clusters to consider while determining the optimal number of clusters.
    """

    def __init__(self, max_clusters):
        self.max_clusters = max_clusters

    def _find_elbow_point(self, feature_vectors):
        """
        Find the optimal number of clusters using the elbow method.

        Parameters
        ----------
        feature_vectors : list of ndarray
            A list of feature vectors.

        Returns
        -------
        elbow_point : int
            The optimal number of clusters.
        """
        within_cluster_sums_of_squares = []
        for i in range(1, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=i, init="k-means++", random_state=0)
            kmeans.fit(feature_vectors)
            within_cluster_sums_of_squares.append(kmeans.inertia_)

        # Find the elbow point
        elbow_point = 0
        max_diff = 0
        for i in range(1, len(within_cluster_sums_of_squares) - 1):
            diff = abs(
                within_cluster_sums_of_squares[i - 1]
                - within_cluster_sums_of_squares[i]
            ) - abs(
                within_cluster_sums_of_squares[i]
                - within_cluster_sums_of_squares[i + 1]
            )
            if diff > max_diff:
                max_diff = diff
                elbow_point = i

        return elbow_point + 1

    def cluster_images(self, image_feature_vector_pairs):
        """
        Cluster images based on their feature vectors.

        Parameters
        ----------
        image_feature_vector_pairs : list of tuple
            A list of tuples containing image and feature vector pairs.

        Returns
        -------
        clustered_images : list of list of tuple
            A list of clusters, where each cluster contains a list of image-feature vector pairs.
        """
        # Extract the feature vectors from the image_feature_vector_pairs
        feature_vectors = [pair[1] for pair in image_feature_vector_pairs]

        # Reshape the feature vectors to have 2 dimensions
        feature_vectors_reshaped = [fv.reshape(-1) for fv in feature_vectors]

        num_clusters = self._find_elbow_point(feature_vectors_reshaped)

        # Cluster the images using KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(
            feature_vectors_reshaped
        )
        labels = kmeans.labels_
        clustered_images = [[] for _ in range(num_clusters)]

        # Assign each image-feature vector pair to the appropriate cluster based on the KMeans labels
        for i, label in enumerate(labels):
            clustered_images[label].append(image_feature_vector_pairs[i])

        # Sort images in each cluster based on their distance to the cluster center
        for cluster_index, cluster in enumerate(clustered_images):
            cluster_center = kmeans.cluster_centers_[cluster_index]
            cluster.sort(
                key=lambda x: np.linalg.norm(x[1].reshape(-1) - cluster_center)
            )

        return clustered_images
