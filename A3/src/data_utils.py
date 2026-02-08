import torch
from torch.nn.utils.rnn import pad_sequence


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            try:
                txt_input = transform(txt_input)
            except:
                txt_input = transform.encode(txt_input).tokens
        return txt_input
    return func


def tensor_transform(token_ids, SOS_IDX, EOS_IDX):
    return torch.cat((torch.tensor([SOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


def make_text_transform(token_transform, vocab_transform, SRC_LANG, TARG_LANG, SOS_IDX, EOS_IDX):
    text_transform = {}
    for ln in [SRC_LANG, TARG_LANG]:
        text_transform[ln] = sequential_transforms(
            token_transform[ln],
            vocab_transform[ln],
            lambda ids: tensor_transform(ids, SOS_IDX, EOS_IDX)
        )
    return text_transform


def make_collate_batch(text_transform, SRC_LANG, TARG_LANG, PAD_IDX):
    def collate_batch(batch):
        src_batch, src_len_batch, trg_batch = [], [], []
        for item in batch:
            processed_text = text_transform[SRC_LANG](item[SRC_LANG])
            src_batch.append(processed_text)
            trg_batch.append(text_transform[TARG_LANG](item[TARG_LANG]))
            src_len_batch.append(processed_text.size(0))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
        trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX, batch_first=True)
        return src_batch, torch.tensor(src_len_batch, dtype=torch.int64), trg_batch
    return collate_batch