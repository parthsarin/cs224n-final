"""
File: roberta.py
----------------

Our implementation of the fine-tuned sliding window model using the RoBERTa
transformer model.
"""

import wandb
import pandas as pd
from nltk import sent_tokenize
import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from peft import LoraConfig, TaskType
from peft import get_peft_model

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
train_data = pd.read_csv("devset.csv")
test_data = pd.read_csv("testset.csv")


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=4,
    lora_alpha=32,
    lora_dropout=0.1,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = get_peft_model(
            RobertaModel.from_pretrained("roberta-base"), peft_config
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=768, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=4),
        )

    def forward(self, x):
        x = self.roberta(x).pooler_output
        x = self.classifier(x)
        x = nn.functional.sigmoid(x)
        return x


model = Model().to("cuda")


def apply_model(model, doc):
    """
    Used at evaluation time to apply the model to a document.
    """
    # break the document into chunks
    sentences = sent_tokenize(doc)
    chunks = []
    curr = ""
    for sentence in sentences:
        proposal = curr + sentence
        proposal_len = len(tokenizer.encode(proposal, add_special_tokens=False))
        if proposal_len > 512:
            chunks.append(curr)
            curr = sentence
        else:
            curr = proposal
    chunks.append(curr)

    # apply the model to each chunk
    chunk_labels = []
    for chunk in chunks:
        tokenized = tokenizer.encode(
            chunk, return_tensors="pt", add_special_tokens=False
        ).to("cuda")
        with torch.no_grad():
            preds = model(tokenized).cpu().detach().numpy()
        chunk_labels.append(preds)
    chunk_labels = np.array(chunk_labels)

    # probability of funding by x is 1 - P(no funding by x)
    # P(no funding by x) = P(no funding by x in c1) * P(no funding by x in c2) * ...
    # P(no funding by x in c1) = 1 - P(funding by x in c1)
    p_no_funding = 1 - chunk_labels
    p_no_funding = np.log(p_no_funding)
    p_no_funding = np.sum(p_no_funding, axis=0)
    p_no_funding = np.exp(p_no_funding)
    p_funding = 1 - p_no_funding
    return p_funding


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train(
    model: nn.Module,
    train_docs: list[str],
    y_train: np.ndarray,
    test_docs: list[str],
    y_test: np.ndarray,
    n_epochs: int = 1_000,
    lr: float = 1e-3,
    save_dir: str = "weights",
):
    """
    Train the model on the given data.
    """
    X_train = [
        tokenizer.encode(doc, return_tensors="pt", add_special_tokens=False)
        for doc in train_docs
    ]

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="roberta-nlp-funding",
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": "sliding RoBERTa",
            "epochs": n_epochs,
        },
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        avg_loss = 0
        for x, labels in zip(X_train, y_train):
            preds = model(x)
            loss = loss_fn(preds, labels)
            avg_loss += loss

        opt.zero_grad()
        avg_loss.backward()
        opt.step()

        avg_loss = avg_loss.item()
        avg_loss /= len(X_train)
        print(f"[epoch {epoch + 1}] loss: {avg_loss:.4f}", end="")

        if epoch % 10 == 0:
            total_correct = 0
            for doc, labels in zip(test_docs, y_test):
                preds = apply_model(model, doc).round()
                acc_vector = 1 - (preds + labels) % 2
                total_correct += sum(acc_vector)

            torch.save(model.state_dict(), f"{save_dir}/model_{epoch}.pt")
            acc = total_correct / (5 * len(test_docs))
            print(f" test accuracy: {acc}", end="")
            wandb.log({"epoch": epoch, "loss": avg_loss, "accuracy": acc})
        else:
            wandb.log({"epoch": epoch, "loss": avg_loss})

        print()


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
