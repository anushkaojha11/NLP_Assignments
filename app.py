# =========================
# Task 3: Similar Word Search Web App (Dash)
# =========================
# Input: a single word
# Output: Top-10 most similar words (dot product)
#
# Uses pretrained embeddings saved as:
#   model/embed_skipgram.pkl
#   model/embed_skipgram_neg.pkl
#   model/embed_glove.pkl
#
# Run:
#   python3 app.py
# Open:
#   http://127.0.0.1:8050

import pickle
import numpy as np

from dash import Dash, dcc, html, Input, Output, State

# -------------------------
# CONFIG
# -------------------------
EMBED_MODEL = "neg"     # "skipgram" | "neg" | "glove"
TOP_K = 10
UNK_TOKEN = "<UNK>"

EMBED_PATHS = {
    "skipgram": "model/embed_skipgram.pkl",
    "neg":      "model/embed_skipgram_neg.pkl",
    "glove":    "model/embed_glove.pkl",
}

# -------------------------
# Load embedding dictionary
# -------------------------
with open(EMBED_PATHS[EMBED_MODEL], "rb") as f:
    EMBED = pickle.load(f)

if UNK_TOKEN not in EMBED:
    raise ValueError("Embedding dict must contain '<UNK>'")

VOCAB = list(EMBED.keys())
EMB_DIM = len(EMBED[UNK_TOKEN])

# Pre-build matrix for fast similarity
EMB_MATRIX = np.stack([EMBED[w] for w in VOCAB], axis=0)  # [V, D]

# -------------------------
# Similarity search
# -------------------------
def get_top_k_similar_words(query_word, k=TOP_K):
    query_word = query_word.lower()

    if query_word not in EMBED:
        query_vec = EMBED[UNK_TOKEN]
    else:
        query_vec = EMBED[query_word]

    scores = EMB_MATRIX @ query_vec   # dot product
    top_idx = np.argsort(-scores)

    results = []
    for idx in top_idx:
        word = VOCAB[idx]
        if word == query_word:
            continue
        results.append(word)
        if len(results) == k:
            break

    return results

# -------------------------
# Dash App
# -------------------------
app = Dash(__name__)
app.title = "Similar Word Search"

PAGE_STYLE = {
    "minHeight": "100vh",
    "background": "linear-gradient(135deg, #f7f7ff, #ffffff)",
    "padding": "40px",
    "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial",
}

CONTAINER_STYLE = {
    "maxWidth": "700px",
    "margin": "0 auto",
}

CARD_STYLE = {
    "border": "1px solid #e6e6e6",
    "borderRadius": "12px",
    "padding": "12px",
    "marginBottom": "8px",
    "backgroundColor": "white",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.05)",
}

app.layout = html.Div(
    style=PAGE_STYLE,
    children=[
        html.Div(
            style=CONTAINER_STYLE,
            children=[
                html.H1("Enter a word to search for similar words"),
                html.P(
                    "The system returns the top 10 most similar words "
                    "using dot product similarity."
                ),

                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr auto", "gap": "10px"},
                    children=[
                        dcc.Input(
                            id="query-input",
                            type="text",
                            placeholder="e.g., election, government, court",
                            style={
                                "padding": "12px",
                                "borderRadius": "10px",
                                "border": "1px solid #ccc",
                                "fontSize": "16px",
                            },
                        ),
                        html.Button(
                            "Search",
                            id="search-btn",
                            n_clicks=0,
                            style={
                                "padding": "12px 18px",
                                "borderRadius": "10px",
                                "border": "none",
                                "backgroundColor": "#2d6cdf",
                                "color": "white",
                                "fontWeight": "600",
                                "cursor": "pointer",
                            },
                        ),
                    ],
                ),

                html.Div(id="results", style={"marginTop": "20px"}),
            ],
        )
    ],
)

@app.callback(
    Output("results", "children"),
    Input("search-btn", "n_clicks"),
    State("query-input", "value"),
)
def update_results(n_clicks, query):
    if not n_clicks:
        return html.Div("Results will appear here.", style={"color": "#666"})

    if not query:
        return html.Div("Please enter a word.", style={"color": "crimson"})

    similar_words = get_top_k_similar_words(query, TOP_K)

    return html.Div(
        children=[
            html.H3(f"Top {TOP_K} similar words"),
            *[
                html.Div(
                    f"{i+1}. {word}",
                    style=CARD_STYLE,
                )
                for i, word in enumerate(similar_words)
            ],
        ]
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)