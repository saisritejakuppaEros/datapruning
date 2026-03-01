import os
import random
import logging
import torch
import numpy as np
import pygtrie
import pickle
from tqdm import tqdm


def get_logger(file_name=None, level=logging.INFO, stdout=True):
    """Logger with optional file and stdout handlers."""
    logger = logging.getLogger("effective_pruning")
    logger.setLevel(level)
    logger.handlers.clear()
    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setLevel(level)
        logger.addHandler(fh)
    if stdout:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        logger.addHandler(sh)
    return logger


def load_arrays(sorted_clusters_path, num_centroids, logger=None):
    """Load cluster_{i}.npy arrays from sorted_clusters_path. Returns list of arrays."""
    result = []
    for i in range(num_centroids):
        p = os.path.join(sorted_clusters_path, f"cluster_{i}.npy")
        result.append(np.load(p))
    return result


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def stringtrie(arr):
    trie = pygtrie.StringTrie()
    i = 0
    for p in tqdm(arr):
        key = p.strip()[:-4]
        trie[key] = 1
        i+=1
    print(f'saved {i} strings')
    return trie

def trie_search(trie, seq='00000/00000'):
    try:
        if trie[seq]:
            return True
    except KeyError:
        return False
    
def build_and_save_trie(data: list, file_path: str):
    print('Building trie structure for the pruned data')
    mytrie = stringtrie(data)
    print('Saving trie.pickle ...')
    with open(file_path, 'wb') as f:
        pickle.dump(mytrie, f)
    print("Saved")
    
    return