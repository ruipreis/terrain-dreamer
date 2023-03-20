import torch
from pathlib import Path

START_TRAIN_AT_IMG_SIZE = 4
DATASET = Path("terrdreamer/dataset/data/final_aw3d30/train_aw3d30")
CHECKPOINT_GENERATOR = "terrdreamer/models/progan_satelite/checkpoints/generator.pth"
CHECKPOINT_DISCRIMINATOR = (
    "terrdreamer/models/progan_satelite/checkpoints/discriminator.pth"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = True
LEARNING_RATE = 1e-3
BATCH_SIZES = [8, 8, 8, 8, 8, 8, 8]
BATCH_SIZE = 8
IMG_CHANNELS = 3
LATENT_SIZE = 256
IN_CHANNELS = 256
DISCRIMINATOR_ITERATIONS = 1
LAMBDA_GP = 10
#PROGRESSIVE_EPOCHS = [5, 6, 7, 10, 15, 20, 30]
#PROGRESSIVE_EPOCHS = [50, 60, 70, 100, 150, 200, 250]
PROGRESSIVE_EPOCHS = [25, 30, 35, 50, 75, 100, 150]
FIXED_NOISE = torch.randn(8, LATENT_SIZE, 1, 1).to(DEVICE)
NUM_WORKERS = 4
