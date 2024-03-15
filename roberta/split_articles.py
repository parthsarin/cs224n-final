from nltk import sent_tokenize
import sys
import csv
from transformers import RobertaTokenizer

csv.field_size_limit(sys.maxsize)

reader = csv.DictReader(open("devset.csv"))
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
labels = ["defense", "corporate", "research_agency", "foundation", "none"]

for article in reader:
    text = article["article_text"]
    sentences = sent_tokenize(text)

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

    article_labels = [article[label] for label in labels]
    article_labels = "\t".join(article_labels)
    print(f'ARTICLE {article["file_id"]}\t\t{article_labels}')

    for i, chunk in enumerate(chunks):
        print(f"CHUNK {i}\t{chunk}")
