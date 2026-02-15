# app.py (Dash) — Sentence Similarity + NLI Prediction
# Run:
#   python3 app.py
# Open:
#   http://127.0.0.1:8050

import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from dash import Dash, html, dcc, Input, Output, State

# Your custom BERT implementation (Task 1) stored in bert_class.py
from bert_class import BERT


# =========================
# 1) Load trained checkpoint
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = "model/sen_bert_full.pth"
checkpoint = torch.load(ckpt_path, map_location=device)

bert_params = checkpoint["bert_params"]
max_seq_length = int(checkpoint.get("max_seq_length", 128))

# Tokenizer for converting text -> input_ids / attention_mask
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Rebuild + load BERT weights
model = BERT(**bert_params, device=device).to(device)
model.load_state_dict(checkpoint["bert_state"])
model.eval()

# Rebuild + load classifier head weights (3-class NLI)
d_model = int(bert_params["d_model"])
classifier_head = nn.Linear(d_model * 3, 3).to(device)
classifier_head.load_state_dict(checkpoint["clf_state"])
classifier_head.eval()

CLASSES = ["entailment", "neutral", "contradiction"]


# =========================
# 2) Helper functions
# =========================
def mean_pool(token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling over non-padding tokens.
    token_embeds: (bs, seq_len, hidden_dim)
    attention_mask: (bs, seq_len) with 1 for real tokens and 0 for padding
    returns: (bs, hidden_dim)
    """
    mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    pooled = torch.sum(token_embeds * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
    return pooled


def predict(sentence_a: str, sentence_b: str):
    """
    Returns:
      - predicted label (entailment/neutral/contradiction)
      - class probabilities (3,)
      - cosine similarity (float)
    """
    inputs_a = tokenizer(
        sentence_a,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
    )
    inputs_b = tokenizer(
        sentence_b,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
    )

    input_ids_a = inputs_a["input_ids"].to(device)
    attn_a = inputs_a["attention_mask"].to(device)

    input_ids_b = inputs_b["input_ids"].to(device)
    attn_b = inputs_b["attention_mask"].to(device)

    bs, seq_len = input_ids_a.shape
    segment_ids = torch.zeros((bs, seq_len), dtype=torch.long, device=device)

    with torch.no_grad():
        # Token-level embeddings from last hidden layer
        u_tok = model.get_last_hidden_state(input_ids_a, segment_ids)
        v_tok = model.get_last_hidden_state(input_ids_b, segment_ids)

        # Sentence embeddings (mean pooled)
        u = mean_pool(u_tok, attn_a)  # (1, d_model)
        v = mean_pool(v_tok, attn_b)  # (1, d_model)

        # SBERT softmax features: [u, v, |u - v|]
        uv_abs = torch.abs(u - v)
        x = torch.cat([u, v, uv_abs], dim=-1)  # (1, 3*d_model)

        # Classifier prediction
        logits = classifier_head(x)  # (1, 3)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # (3,)
        pred_idx = int(np.argmax(probs))
        pred_label = CLASSES[pred_idx]

        # Cosine similarity (extra interpretability)
        cos_sim = float(cosine_similarity(u.cpu().numpy(), v.cpu().numpy())[0, 0])

    return pred_label, probs, cos_sim


# =========================
# 3) Dash app + styling
# =========================
app = Dash(__name__)
app.title = "Sentence Similarity & NLI"

# Load Poppins from Google Fonts and apply global styling
app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>Sentence Similarity & NLI</title>
    {%favicon%}
    {%css%}
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
      body {
        font-family: 'Poppins', sans-serif;
        background: #f5f7fa;
        margin: 0;
        padding: 0;
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""

app.layout = html.Div(
    style={
        "maxWidth": "860px",
        "margin": "50px auto",
        "padding": "28px",
        "backgroundColor": "white",
        "borderRadius": "14px",
        "boxShadow": "0px 6px 24px rgba(0,0,0,0.08)",
    },
    children=[
        html.H2("Sentence Similarity & NLI Prediction", style={"textAlign": "center", "marginBottom": "6px"}),
        html.P(
            "Enter two sentences and click Predict to see the predicted relationship and cosine similarity.",
            style={"textAlign": "center", "color": "#4b5563", "marginTop": "0px"},
        ),

        html.Div(
            style={"display": "flex", "gap": "16px", "marginTop": "18px"},
            children=[
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.Label("Sentence A", style={"fontWeight": "500"}),
                        dcc.Textarea(
                            id="sentence-a",
                            value="A man is drinking juice.",
                            style={"width": "100%", "height": "120px", "marginTop": "6px"},
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.Label("Sentence B", style={"fontWeight": "500"}),
                        dcc.Textarea(
                            id="sentence-b",
                            value="An older man is drinking orange juice at a restaurant.",
                            style={"width": "100%", "height": "120px", "marginTop": "6px"},
                        ),
                    ],
                ),
            ],
        ),

        html.Div(
            style={"display": "flex", "justifyContent": "center", "marginTop": "18px"},
            children=[
                html.Button(
                    "Predict",
                    id="predict-btn",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#2563eb",   # blue button
                        "color": "white",
                        "padding": "10px 22px",
                        "border": "none",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                        "fontWeight": "600",
                        "fontSize": "14px",
                    },
                )
            ],
        ),

        html.Hr(style={"marginTop": "20px", "marginBottom": "16px"}),

        html.Div(id="result-block"),
    ],
)


# =========================
# 4) Callback
# =========================
@app.callback(
    Output("result-block", "children"),
    Input("predict-btn", "n_clicks"),
    State("sentence-a", "value"),
    State("sentence-b", "value"),
)
def on_predict(n_clicks, sent_a, sent_b):
    if n_clicks == 0:
        return ""

    if not sent_a or not sent_a.strip() or not sent_b or not sent_b.strip():
        return html.Div(
            "Please enter both sentences.",
            style={"color": "#b91c1c", "fontWeight": "500"},
        )

    label, probs, sim = predict(sent_a.strip(), sent_b.strip())

    # Get highest probability
    max_prob = float(np.max(probs))

    return html.Div(
        style={
            "padding": "16px",
            "borderRadius": "10px",
            "backgroundColor": "#f8fafc",
            "border": "1px solid #e5e7eb",
            "textAlign": "center"
        },
        children=[
            html.H3(f"Prediction: {label}"),
            html.P(f"Confidence: {max_prob:.4f}"),
            html.P(f"Cosine Similarity: {sim:.4f}")
        ],
    )


# =========================
# 5) Run server
# =========================
if __name__ == "__main__":
    app.run(debug=True)