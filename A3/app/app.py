import os, sys
import torch
import torchtext

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd

from torchtext.data.utils import get_tokenizer
from nepalitokenizers import WordPiece

from src.model_def import build_model
from src.data_utils import make_text_transform
from src.infer import greedy_decode


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SRC_LANG = "en"
TARG_LANG = "ne"

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

vocab_transform = torch.load("model/vocab.pt", map_location="cpu")

token_transform = {}
token_transform[SRC_LANG] = get_tokenizer("spacy", language="en_core_web_sm")
token_transform[TARG_LANG] = WordPiece()

text_transform = make_text_transform(
    token_transform=token_transform,
    vocab_transform=vocab_transform,
    SRC_LANG=SRC_LANG,
    TARG_LANG=TARG_LANG,
    SOS_IDX=SOS_IDX,
    EOS_IDX=EOS_IDX,
)

config = {
    "HID_DIM": 256,
    "ENC_LAYERS": 3,
    "DEC_LAYERS": 3,
    "ENC_HEADS": 8,
    "DEC_HEADS": 8,
    "ENC_PF_DIM": 512,
    "DEC_PF_DIM": 512,
    "ENC_DROPOUT": 0.1,
    "DEC_DROPOUT": 0.1,
    "ATTEN_TYPE": "additive",
    "MAX_LEN": 5000,   # must match the checkpoint (pos_embedding sizes)
    "SRC_PAD_IDX": PAD_IDX,
    "TRG_PAD_IDX": PAD_IDX,
}

INPUT_DIM = len(vocab_transform[SRC_LANG])
OUTPUT_DIM = len(vocab_transform[TARG_LANG])

model = build_model(config, INPUT_DIM, OUTPUT_DIM, DEVICE)
state_dict = torch.load("model/additive_state_dict.pt", map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

ne_itos = vocab_transform[TARG_LANG].get_itos()


def decode_ids_to_tokens(pred_ids, eos_idx=EOS_IDX, max_out_tokens=40):
    """
    Robust decoding:
    - stops by EOS *id* (safer than string)
    - limits max output tokens for UI
    - breaks simple repetition loops
    """
    tokens = []
    ids = pred_ids.tolist()

    for t in ids:
        if t == eos_idx:
            break
        if t in (SOS_IDX, PAD_IDX):  # optionally skip UNK as well
            continue

        tok = ne_itos[t]
        tokens.append(tok)

        # hard cap for UI
        if len(tokens) >= max_out_tokens:
            break

        # repetition guard (prevents long looping outputs)
        if len(tokens) >= 12 and len(set(tokens[-8:])) <= 2:
            break

    return tokens


app = Dash(__name__)
app.layout = html.Div(
    style={"maxWidth": "900px", "margin": "40px auto", "fontFamily": "system-ui"},
    children=[
        html.H2("English to Nepali Translation"),
        dcc.Textarea(
            id="src_text",
            value="Hello. How are you?",
            style={"width": "100%", "height": "100px"},
        ),
        html.Br(),
        html.Button("Translate", id="btn", n_clicks=0),
        html.Hr(),
        html.H4("Output"),
        html.Div(id="out_text", style={"whiteSpace": "pre-wrap", "fontSize": "18px"}),
        html.Hr(),
        html.H4("Attention Heatmap (head 0)"),
        dcc.Graph(id="attn_fig"),
    ],
)


@app.callback(
    Output("out_text", "children"),
    Output("attn_fig", "figure"),
    Input("btn", "n_clicks"),
    State("src_text", "value"),
)
def translate(n_clicks, src_text):
    if not src_text or not src_text.strip():
        return "", px.imshow([[0]])

    src_ids = text_transform[SRC_LANG](src_text.lower())
    src_tensor = src_ids.unsqueeze(0)

    # keep decode length reasonable (prevents super long output)
    pred_ids, attn = greedy_decode(
        model,
        src_tensor,
        SOS_IDX,
        EOS_IDX,
        max_len=80,   # internal decode steps
    )

    # convert ids -> tokens safely
    tokens = decode_ids_to_tokens(pred_ids, eos_idx=EOS_IDX, max_out_tokens=40)
    out = " ".join(tokens)

    if attn is None:
        return out, px.imshow([[0]])

    # attention: (T, S) expected for head 0
    A = attn[0, 0].detach().cpu().numpy()

    src_tokens = ["<sos>"] + token_transform[SRC_LANG](src_text.lower()) + ["<eos>"]
    trg_tokens = ["<sos>"] + [ne_itos[t] for t in pred_ids.tolist()[1:]]  # exclude first SOS id

    t_len = min(A.shape[0], len(trg_tokens))
    s_len = min(A.shape[1], len(src_tokens))

    df = pd.DataFrame(
        A[:t_len, :s_len],
        index=trg_tokens[:t_len],
        columns=src_tokens[:s_len],
    )
    fig = px.imshow(df, aspect="auto")

    return out, fig


if __name__ == "__main__":
    app.run(debug=True)