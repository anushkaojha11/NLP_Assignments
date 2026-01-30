from dash import Dash, html, dcc, Input, Output, State
import torch
import pickle
import os
from torchtext.data.utils import get_tokenizer
from lstmmodel import LSTMLanguageModel

# -------------------------
# Device configuration
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Tokenizer
# -------------------------
tokenizer = get_tokenizer("basic_english")

# -------------------------
# Paths
# -------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "model")

vocab_path = os.path.join(model_dir, "vocab_lm.pkl")
weights_path = os.path.join(model_dir, "best-val-lstm_lm.pt")

# -------------------------
# Load vocabulary
# -------------------------
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

# -------------------------
# Load trained model
# -------------------------
vocab_size = len(vocab)
emb_dim = 256
hid_dim = 256
num_layers = 2
dropout_rate = 0.3

model = LSTMLanguageModel(
    vocab_size, emb_dim, hid_dim, num_layers, dropout_rate
).to(device)

model.load_state_dict(torch.load(weights_path, map_location=device))

# -------------------------
# Text generation function
# -------------------------
def generate_text(prompt, max_len, temperature):
    model.eval()

    tokens = tokenizer(prompt)
    if len(tokens) == 0:
        tokens = ["<unk>"]

    indices = [vocab[t] for t in tokens]
    hidden = model.init_hidden(1, device)

    with torch.no_grad():
        for _ in range(max_len):
            src = torch.LongTensor([[indices[-1]]]).to(device)
            output, hidden = model(src, hidden)

            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, 1).item()

            if next_idx == vocab["<eos>"]:
                break

            indices.append(next_idx)

    itos = vocab.get_itos()
    return " ".join([itos[i] for i in indices])

# -------------------------
# Dash App
# -------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Text Generation using LSTM Language Model",
            style={"textAlign": "center"}),

    dcc.Input(
        id="prompt-input",
        type="text",
        placeholder="Enter a text prompt...",
        style={"width": "60%", "margin": "20px auto", "display": "block"}
    ),

    html.Button(
        "Generate",
        id="generate-btn",
        n_clicks=0,
        style={"display": "block", "margin": "10px auto"}
    ),

    html.Div(id="output-text",
             style={"margin": "30px", "fontSize": "18px"})
])

# -------------------------
# Callback
# -------------------------
@app.callback(
    Output("output-text", "children"),
    Input("generate-btn", "n_clicks"),
    State("prompt-input", "value")
)
def generate(n_clicks, prompt):
    if n_clicks == 0 or not prompt:
        return "Enter a prompt to generate text."

    temperatures = [0.1, 0.5, 0.7, 0.9, 1.0]

    outputs = []
    for temp in temperatures:
        text = generate_text(
            prompt=prompt.strip(),
            max_len=50,          # generate more words
            temperature=temp
        )

        outputs.append(
            html.Div([
                html.H4(f"Temperature: {temp}"),
                html.P(text)
            ], style={
                "marginBottom": "20px",
                "padding": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "6px",
                "backgroundColor": "#f9f9f9"
            })
        )

    return outputs

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)