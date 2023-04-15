import torch

DEVICE = torch.device('cpu')

MODEL_PARAMETERS = {
    'default_model': {
        'mf_dim': 8,
        'layers': [64, 32, 16, 8],
        'reg_layers': [0, 0, 0, 0],
        'reg_mf': 0
    },
    'FedNCF': {
        'mf_dim': 16,
        'layers': [64, 32, 16, 8],
        'reg_layers': [0, 0, 0, 0],
        'reg_mf': 0
    },
}

BATCH_SIZE = 64
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.001
NUM_NEGATIVES = 4
