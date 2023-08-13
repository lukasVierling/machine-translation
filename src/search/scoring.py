import sys
sys.path.append('..')

import subprocess

from src.preprocessing.dictionary import END
from src.data_loader import DataLoader
from src.pickle_loader import PickleLoader
from data.dataset import Dataset_FFN
#from utils.alignment import monotonic_alignment

from torch.utils.data import DataLoader as TorchDataLoader
from torch.nn.functional import log_softmax
import torch


def score_corpus(source_path, target_path, model):
    """
    Calculate score of a corpus based on a given model, dictionary, encoder and alignment

    Parameters:
        source_path (str): Path to source corpus (indexed)
        target_path (str): Path to target corpus (indexed)
        model (nn.Module): Model to use for scoring

    Returns:
        score (List[float]): Log score of each sentence in the corpus
    """
    # load parameters
    model_config = model.config
    bpe_ops = model_config['bpe_ops']
    joined_bpe = model_config['joined_bpe']
    window_size = model_config['window_size']
    alignment = model_config['batch_alignment']

    # torch data loader for the source and target
    torch_dataset = Dataset_FFN(source_path, target_path, is_indexed=True, bpe_ops=bpe_ops, joined_bpe=joined_bpe, window_size=window_size, batch_alignment=alignment)
    torch_dl = TorchDataLoader(torch_dataset, batch_size=1, shuffle=False)

    # Loop over source and predict target probability distribution
    # Sum the log probabilities of the target, based on given alignment
    scores = []
    score = 0
    for i, (source, target, label) in enumerate(torch_dl):
        model.eval()
        with torch.no_grad():
            output = model(source, target)
            score += log_softmax(output, dim=1)[0][label[0]].item()

        if label[0].item() == END:
            scores.append(score)
            score = 0
    
    return scores


if __name__ == "__main__":
    # load model
    model = torch.load('../logs/gutere_modell/checkpoints/model_6.pt')


    # load data
    target = "../data/7k_BPE_indexed/dev/target.dev.pickle"
    source = "../data/7k_BPE_indexed/dev/source.dev.pickle"

    # score corpus
    scores = score_corpus(source, target, model, "uniform")

    # print scores
    print(scores)
    print(f"Len scores: {len(scores)}")

    # load data
    target_dings = PickleLoader.load(source)
    print(f"Len sentences: {len(target_dings)}")
    # min score
    print(f"Max score: {max(scores)}")
