import wandb
import pandas as pd
from nltk import sent_tokenize
import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

data = pd.read_csv("devset.csv")
test_data = pd.read_csv("testset.csv")

data = data[
    ["article_text", "military", "corporate", "research_agency", "foundation", "none"]
]
test_data = test_data[
    ["article_text", "military", "corporate", "research_agency", "foundation", "none"]
]


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

from peft import LoraConfig, TaskType
from peft import get_peft_model

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


def apply_model(doc):
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


# file_id, article_text, military, corporate, research_agency, foundation, none
X_train = list(data["article_text"])
y_train = torch.Tensor(
    data[["military", "corporate", "research_agency", "foundation", "none"]].to_numpy()
).cuda()

X_val = list(test_data["article_text"])
y_val = test_data[
    ["military", "corporate", "research_agency", "foundation", "none"]
].to_numpy()

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,stratify=y)
X_train_tokenized = [
    tokenizer.encode(article, return_tensors="pt").to("cuda") for article in X_train
]
X_val_tokenized = [
    tokenizer.encode(article, return_tensors="pt").to("cuda") for article in X_val
]

opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="roberta-nlp-funding",
    # track hyperparameters and run metadata
    config={"learning_rate": 0.001, "architecture": "pooled RoBERTa", "epochs": 1000},
)

for epoch in range(1_000):
    avg_loss = 0
    for x, labels in zip(X_train_tokenized, y_train):
        preds = model(x)
        loss = loss_fn(preds, labels)
        avg_loss += loss

    opt.zero_grad()
    avg_loss.backward()
    opt.step()

    avg_loss = avg_loss.item()
    avg_loss /= len(X_train_tokenized)
    print(f"[epoch {epoch + 1}] loss: {avg_loss:.4f}", end="")

    if epoch % 10 == 0:
        total_correct = 0
        for x, labels in zip(X_val_tokenized, y_val):
            with torch.no_grad():
                preds = model(x).cpu().detach().numpy().round()
            acc_vector = 1 - (preds + labels) % 2
            total_correct += sum(acc_vector)

        torch.save(model.state_dict(), f"weights/model_{epoch}.pt")
        acc = total_correct / (5 * len(X_val_tokenized))
        print(f" test accuracy: {acc}", end="")
        wandb.log({"epoch": epoch, "loss": avg_loss, "accuracy": acc})
    else:
        wandb.log({"epoch": epoch, "loss": avg_loss})

    print()

total_correct = 0
for x, labels in zip(X_val_tokenized, y_val):
    with torch.no_grad():
        preds = model(x).cpu().detach().numpy().round()
    acc_vector = 1 - (preds + labels) % 2
    print("model predicted: ", preds)
    print("correct answer: ", labels)
    print("correctness: ", acc_vector)
    print()
