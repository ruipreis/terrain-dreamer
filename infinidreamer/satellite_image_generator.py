import torch


class SatelliteImageGenerator:
    """
    A class to simulate satellite image generation with feature extraction
    using a ProGAN model.

    Attributes
    ----------
    progan_model : object
        The ProGAN model used for image generation.
    land_cover_colors : list
        List of RGB color tuples representing different land cover types.

    Methods
    -------
    generate_images_with_features(batch_size, device):
        Generates a list of images and their corresponding feature vectors.
    """

    def __init__(self, progan_model, device):
        """
        Constructs a SatelliteImageGenerator object.

        Parameters
        ----------
        progan_model : object
            The ProGAN model used for image generation.
        device : str
            The device to be used for image generation (CPU or GPU).
        """
        self.model = progan_model
        self.device = device

    def generate_images_with_features(
        self, batch_size, similarity_weight=0.6, diversity_weight=0.4
    ):
        """
        Generates a list of images and their corresponding feature vectors
        using the ProGAN model.

        Parameters
        ----------
        batch_size : int
            The number of images to generate.
        similarity_weight : float, optional
            Weight assigned to similarity between the generated images (default is 0.6).
        diversity_weight : float, optional
            Weight assigned to diversity between the generated images (default is 0.4).

        Returns
        -------
        image_feature_vector_pairs : list
            A list of tuples containing the generated images and their corresponding feature vectors.
        """

        # Create random noise input for the model
        random_noise = [
            torch.randn(1, 256, 1, 1) * diversity_weight
            + torch.randn(1, 256, 1, 1) * similarity_weight
            for _ in range(batch_size)
        ]
        random_noise = torch.cat(random_noise, dim=0).to(self.device)

        # Generate satellite images using the model
        satellite_images = self.model(random_noise)

        # Generate a list of images paired with their feature vectors
        image_feature_vector_pairs = []

        for i in range(satellite_images.size(0)):
            image = satellite_images[i]
            feature_vector = random_noise[i]

            # Normalize the image to a range of [0, 1] for saving or further processing
            image_normalized = (image + 1) / 2

            # Convert the image to a numpy array with shape (H, W, C)
            image_np = image_normalized.permute(1, 2, 0).cpu().detach().numpy()

            # Convert the feature vector to a numpy array
            feature_vector_np = feature_vector.cpu().detach().numpy()

            # Add the image and feature vector pair to the list
            image_feature_vector_pairs.append((image_np, feature_vector_np))

        return image_feature_vector_pairs
