import cv2
import torch
import torch.nn.functional as F
from qdrant_client.models import PointStruct
from torchvision.models import ResNet50_Weights, resnet50

from terrdreamer.grid.run import get_client, recreate_collection
from terrdreamer.models.pretrained import PretrainedProGAN


def tensor_transform(input_tensor):
    # Resize
    resized_tensor = F.interpolate(
        input_tensor, size=(256, 256), mode="bilinear", align_corners=False
    )

    # CenterCrop
    def center_crop(tensor, output_size):
        h, w = tensor.shape[-2:]
        th, tw = output_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return tensor[..., i : i + th, j : j + tw]

    cropped_tensor = center_crop(resized_tensor, (224, 224))

    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=input_tensor.device).view(
        1, 3, 1, 1
    )
    std = torch.tensor([0.229, 0.224, 0.225], device=input_tensor.device).view(
        1, 3, 1, 1
    )
    normalized_tensor = (cropped_tensor - mean) / std

    return normalized_tensor


class FeatureVectorExtractor:
    def __init__(self):
        # Load the pre-trained ResNet-18 model
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()

    def get_dim(self):
        return 2048

    def __call__(self, img):
        img = tensor_transform(img)
        vecs = self.model(img)
        vecs = vecs.view(vecs.size(0), -1)

        # Normalize the vectors
        vecs = F.normalize(vecs, p=2, dim=1)

        return vecs

    def to(self, device):
        self.model.to(device)
        return self


def random_sample(num_samples, device: str = "cuda"):
    return torch.randn(num_samples, 256, 1, 1, device=device)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-batches", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--collection-name", type=str, default="satelite_grids")
    parser.add_argument("--image-dir", type=Path, default="data/satelite_images")
    args = parser.parse_args()

    image_dir = args.image_dir
    collection_name = args.collection_name
    batch_size = args.batch_size
    n_batches = args.n_batches

    image_dir.mkdir(exist_ok=True, parents=True)

    progan_model = PretrainedProGAN()
    progan_model = progan_model.to("cuda")

    # Load a feature extractor
    feature_extractor = FeatureVectorExtractor()
    feature_extractor = feature_extractor.to("cuda")

    # Instantiate a Qdrant client and recreate the collection
    client = get_client()
    recreate_collection(client, collection_name, args.dim)

    # Keep track of the number of vectors added
    n_vectors = 0

    for _ in range(n_batches):
        # Generate a batch of random satelite
        rand_sat = progan_model(random_sample(batch_size))

        # The output of the model is in the range [-1, 1], need to convert to [0, 255]
        rand_sat = (rand_sat + 1) * 127.5

        # Extract the feature vectors
        feature_vectors = feature_extractor(rand_sat).detach().cpu()
        rand_sat = rand_sat.detach().cpu()

        # Add the vectors to the collection
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=idx,
                    vector=feature_vectors[idx % batch_size].tolist(),
                )
                for idx in range(n_vectors, n_vectors + batch_size)
            ],
        )

        # Save the images
        for idx in range(n_vectors, n_vectors + batch_size):
            sat_img = rand_sat[idx % batch_size].numpy().transpose(1, 2, 0)

            # Turn the image into bgr
            sat_img = sat_img[:, :, ::-1]

            cv2.imwrite(str(image_dir / f"{idx}.png"), sat_img)

        n_vectors += batch_size

    print("Finished adding a bunch of random vectors")
