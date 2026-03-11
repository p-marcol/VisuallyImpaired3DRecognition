import torch

IMG_SIZE=224
BATCH_SIZE=8
EPOCHS=5
LR = 1e-3 
NUM_CLASSES = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'