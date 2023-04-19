import os
import gdown
import pkg_resources
import torch
import torch.nn as nn
from terrdreamer.models.progan_satelite.progan import Generator
from terrdreamer.models.infinity_grid import DeepFillV1

def download_from_google_drive(file_id, output_path):
    gdown.download(id=file_id, output=output_path, quiet=False)

def check_and_download_pretrained_model(model_base_name, file_id):
    resource_path = f"models/pretrained/{model_base_name}"

    # Attempt to find the resource
    res_path = pkg_resources.resource_filename("terrdreamer", resource_path)

    if os.path.isfile(res_path):
        print(f"Using locally cached '{model_base_name}' model")
    else:
        # If the resource is not found, download it
        print(
            f"Model '{model_base_name}' not found in the '{res_path}' subpackage. Downloading..."
        )
        download_from_google_drive(file_id, res_path)
        print(f"Model '{model_base_name}' downloaded to '{res_path}'.")

    return res_path

class PretrainedProGAN(nn.Module):
    GENERATOR_FILE_NAME = "progan_generator.pth"
    FILE_ID = "1okMqM3D35wuFk4ZznYycYEVVF95uyJV2"

    def __init__(self):
        super().__init__()
        self.model_path = check_and_download_pretrained_model(
            self.GENERATOR_FILE_NAME, self.FILE_ID
        )

        # Create the model
        self.progan = Generator(256, 256, 3)

        # Load the pretrained model
        self.progan.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))

        # Set the model to evaluation mode
        self.progan.eval()

    def forward(self, x):
        # Alpha = 1, n_steps = 7 - 1
        return self.progan(x, 1, 6)

class PretrainedDeepFillV1:
    GENERATOR_FILE_NAME = "deepfillv1_generator.pth"
    FILE_ID = "16lStlTfhLSNsGFAmRvAWoXaLssGHFobu"

    def __init__(self):
        self.model_path = check_and_download_pretrained_model(
            self.GENERATOR_FILE_NAME, self.FILE_ID
        )

        # Create the model
        self.deepfillv1 = DeepFillV1(inference=True)

        # Load the pretrained model
        self.deepfillv1.inpaint_generator.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))

        # Set the inpaint_generator to not require gradient computation
        self.deepfillv1.inpaint_generator.set_requires_grad(False)

    def __call__(self, masked_x, mask):
        return self.deepfillv1.eval(masked_x, mask)

    def to(self, device):
        self.deepfillv1.inpaint_generator.to(device)
        return self
