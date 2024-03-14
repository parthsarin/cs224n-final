"""
File: annotate.py
-----------------
"""
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from json import load, dump
import numpy as np
from glob import glob
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

load_dotenv()

N_AGENCIES = 5


# ------------------------------------------------------------------------------
# generation utilities
# ------------------------------------------------------------------------------
class ArticleClassifier:
    def annotate(self, article, prompt=None):
        raise NotImplementedError("annotate method must be implemented")


class OpenAIClassifier(ArticleClassifier):
    def __init__(
        self,
        model,
        prompt_file="prompts/annotate-openai.txt",
        max_tokens=500,
        temperature=0.1,
        reprompt=True,
    ):
        self.client = OpenAI()
        self.model = model
        self.prompt_template = open(prompt_file, "r").read()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reprompt = reprompt

    def generate(self, messages):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            n=1,
            logprobs=True,
            top_logprobs=2,
        )

        # get the reasoning
        r = r.choices[0]
        msg = r.message.content.strip()
        reasoning = msg.split("\n")[0].strip()

        # get the logprobs
        msg = msg.split("\n")[1:]
        raw_lps = r.logprobs.content
        for idx, lp in enumerate(raw_lps):
            # ignore the first line
            if "\n" in lp.token:
                break
        lps = raw_lps[idx:]
        lps = [lp for lp in lps if lp.token in (" yes", " no")]
        agencies = [a.split(":")[0].lower() for a in msg if a.strip()]

        return r.message.content, reasoning, lps, agencies

    def annotate(self, article, prompt=None):
        if prompt is None:
            prompt = self.prompt_template
        prompt = prompt.format(article=article)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant and an expert in the history of natural langauge processing.",
            },
            {"role": "user", "content": prompt},
        ]

        # generate the response
        msg, reasoning, lps, agencies = self.generate(messages)

        # if we need to reprompt
        if (len(agencies) != N_AGENCIES or len(lps) != N_AGENCIES) and self.reprompt:
            messages.extend(
                [
                    {"role": "assistant", "content": msg},
                    {
                        "role": "user",
                        "content": f"That response was incorrectly formatted. You included {len(agencies)} agencies and {len(lps)} yes/no statements. Both should be {N_AGENCIES}. Please try again. Make sure your response includes exactly six lines, with the reasoning on the first line, and no blank lines.",
                    },
                ]
            )

            msg, reasoning, lps, agencies = self.generate(messages)

        # still didn't work
        if len(lps) != N_AGENCIES or len(agencies) != N_AGENCIES:
            raise ValueError(f"Incorrectly formatted response: {msg}")

        # get probabilities
        probs = []
        for lp in lps:
            if lp.token == " yes":
                probs.append(np.exp(lp.logprob))

            if lp.token == " no":
                probs.append(1 - np.exp(lp.logprob))

        return dict(zip(agencies, probs)), reasoning

    def __repr__(self):
        return f'OpenAIClassifier(model="{self.model}")'


class HuggingFaceClassifier(ArticleClassifier):
    def __init__(
        self,
        model,
        prompt_file="prompts/annotate-hf.txt",
        question_file="prompts/questions.json",
        max_tokens=500,
        temperature=0.1,
    ):
        self.model_str = model
        self.model = AutoModelForCausalLM.from_pretrained(
            model, load_in_4bit=True, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.prompt_template = open(prompt_file, "r").read()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.questions = load(open(question_file, "r"))

    def __repr__(self):
        return f'HuggingFaceClassifier(model="{self.model_str}")'

    def annotate(self, article, prompt=None):
        if prompt is None:
            prompt = self.prompt_template
        prompt = prompt.format(article=article).strip()

        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=False
        ).cuda()

        with torch.cuda.amp.autocast():
            output = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # get the logprobs
            logprobs = self.model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            )
            logprobs = logprobs[0].cpu().detach().numpy()
            probs = np.exp(logprobs)

        # remove the prompt
        output = output.sequences[:, input_ids.shape[1] :][0]
        generation = self.tokenizer.decode(output).strip()
        breakpoint()
        reasoning = f'Reasoning: {generation.split("\n")[0].strip()}'
        answer = "\n".join(generation.split("\n")[1:])

        # can't parse the answer
        if ("yes" not in answer) and ("no" not in answer):
            return None, None

        # get the probabilities
        probs = []
        capture_ans = False
        for tok, score, idx in zip(output, probs, range(len(output))):
            tok = self.tokenizer.decode(tok).strip().lower()
            if "\n" in tok:
                capture_ans = True

            if tok in ("yes", "no") and capture_ans:
                p = score if tok == "yes" else 1 - score
                probs.append(p.item())

        # get the labels
        labels = [t[: t.find(":")] for t in answer.split("\n") if t.strip()]
        out = dict(zip(labels, probs))

        return out, reasoning


# ------------------------------------------------------------------------------
# interface
# ------------------------------------------------------------------------------
def main(args):
    to_annotate = glob(f"{args.in_dir}/*.json")
    out_dir = args.out_dir if args.out_dir else args.in_dir

    if args.model.startswith("gpt-"):
        model = OpenAIClassifier(
            model=args.model, temperature=args.temperature, reprompt=args.reprompt
        )

    else:
        model = HuggingFaceClassifier(model=args.model, temperature=args.temperature)

    with tqdm(total=len(to_annotate)) as pbar:
        for path in to_annotate:
            filename = path.split("/")[-1]
            data = load(open(path, "r"))

            # if already annotated, skip
            if "funding" in data and not args.reannotate:
                pbar.write(f"File {path} already annotated")
                pbar.update(1)
                continue

            # get the article
            try:
                article = data["article"]
            except KeyError:
                pbar.write(f"No article found in file {path}")
                pbar.update(1)
                continue

            # annotate the article
            try:
                funding, reasoning = model.annotate(article)
            except ValueError as e:
                pbar.write(f"Error annotating file {path}: {e}")
                pbar.update(1)
                continue
            data = {**data, "funding": funding, "reasoning": reasoning}

            # save the annotated article
            dump(data, open(f"{out_dir}/{filename}", "w"), indent=4)

            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="annotate JSON files representing academic articles with agency funding probabilities"
    )

    parser.add_argument(
        "in_dir", type=str, help="input directory containing JSON files"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4",
        help="the AI model to use -- gpt-* will be passed to OpenAI, otherwise HuggingFace",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="",
        help="output directory for annotated JSON files (default is the input directory)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.1,
        help="temperature for generation",
    )
    parser.add_argument(
        "-r",
        "--reprompt",
        action="store_true",
        help="reprompt if the response is not formatted correctly",
    )
    parser.add_argument(
        "-a",
        "--reannotate",
        action="store_true",
        help="reannotate files that have already been annotated",
    )

    args = parser.parse_args()
    main(args)
