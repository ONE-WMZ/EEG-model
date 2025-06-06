import random
import numpy as np
import torch

""" 随机种子 """
def set_seed(seed):
    # Python's built-in random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Sets the seed for generating random numbers on all GPUs.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # This can slow down training.

"""
    固定随机种子：
        train_loader = torch.utils.data.DataLoader( worker_init_fn=set_seed(42)
                                                    )
"""