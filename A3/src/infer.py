import torch


def greedy_decode(model, src_tensor, SOS_IDX, EOS_IDX, max_len=80):
    model.eval()
    device = next(model.parameters()).device

    src_tensor = src_tensor.to(device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    ys = torch.tensor([[SOS_IDX]], device=device)

    attentions = None

    for _ in range(max_len):
        trg_mask = model.make_trg_mask(ys)
        with torch.no_grad():
            out, attn = model.decoder(ys, enc_src, trg_mask, src_mask)

        next_token = out[:, -1, :].argmax(dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
        attentions = attn

        if next_token == EOS_IDX:
            break

    return ys.squeeze(0), attentions