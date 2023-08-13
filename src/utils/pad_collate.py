import sys
sys.path.append('..')

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
from src.preprocessing.dictionary import PADDING

def pad_collate(batch):
    """
    Sources:
    https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
    https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
    https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence
    https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html#torch.nn.utils.rnn.pad_packed_sequence
    https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_sequence.html#torch.nn.utils.rnn.pack_sequence    """
    (xx, yy) = zip(*batch)
    x_lens = torch.tensor([len(x) for x in xx])
    y_lens = torch.tensor([len(y) for y in yy])

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=PADDING)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=PADDING)

    return {'src': xx_pad, 'trg': yy_pad, 'src_lens': x_lens, 'trg_lens': y_lens, 'labels': yy_pad[:, 1:]}