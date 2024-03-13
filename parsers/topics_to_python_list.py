"""
File: topics_to_python_list.py
------------------------------

Uses Mistral to parse the topics as a Python list from the data that Vyoma shared.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import sys
import argparse
from tqdm import tqdm
from json import dump

csv.field_size_limit(sys.maxsize)

PROMPT = """\
[INST]
Parse the Python list from this output, if it exists, or return an empty list otherwise:
I apologize, but I cannot find any explicit references to practicality in this paper.
[/INST]

[ANS]
[]
[/ANS]

[INST]
Parse the Python list from this output, if it exists, or return an empty list otherwise:
- In general, we find that the median score is highest for β-TCVAE and it is close to the highest score
achieved by all methods. Despite the best half of the β-TCVAE runs achieving relatively high scores,
we see that the other half can still perform poorly. Low-score outliers exist in the 3D faces dataset,
although their scores are still higher than the median scores achieved by both VAE and InfoGAN.
- We find that β-TCVAE provides a better trade-off between density estimation and disentanglement.
Notably, with higher values of β, the mutual information penality in β-VAE is too strong and this
hinders the usefulness of the latent variables. However, β-TCVAE with higher values of β consistently
results in models with higher disentanglement score relative to β-VAE.
[/INST]

[ANS]
[
    "In general, we find that the median score is highest for β-TCVAE and it is close to the highest score achieved by all methods. Despite the best half of the β-TCVAE runs achieving relatively high scores, we see that the other half can still perform poorly. Low-score outliers exist in the 3D faces dataset, although their scores are still higher than the median scores achieved by both VAE and InfoGAN.",
    "We find that β-TCVAE provides a better trade-off between density estimation and disentanglement. Notably, with higher values of β, the mutual information penality in β-VAE is too strong and this hinders the usefulness of the latent variables. However, β-TCVAE with higher values of β consistently results in models with higher disentanglement score relative to β-VAE."
]
[/ANS]

[INST]
Parse the Python list from this output, if it exists, or return an empty list otherwise:
{model_output}
[/INST]

[ANS]
"""


class HFParser:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)

    def parse(self, model_output):
        prompt = PROMPT.format(model_output=model_output)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=len(model_output) + 100,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            num_return_sequences=1,
        )
        out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # remove prompt
        out = out[len(prompt) :]

        # find the [/ANS] tag
        start = out.find("[/ANS]")
        if start == -1:
            raise ValueError("No [/ANS] tag found in the model output")

        # trim the output up to the [/ANS] tag
        out = out[:start].strip()

        return eval(out)


def main(args):
    model = HFParser(args.model_name)

    # count number of lines
    reader = csv.DictReader(open(args.input_file))
    n_lines = sum(1 for row in reader)

    # process files
    reader = csv.DictReader(open(args.input_file))
    out = []
    with tqdm(total=n_lines) as pbar:
        for row in reader:
            try:
                response = model.parse(row["response"])
                out.append({"title": row["Title"], "response": response})
            except Exception as e:
                pbar.write(f"Error parsing response for {row['Title']}: {e}")
            pbar.update(1)

            # write to the outfile
            dump(out, open(args.output_file, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse the topics as a Python list from the data that Vyoma shared."
    )
    parser.add_argument(
        "model_name", type=str, help="The name of the model to use for parsing"
    )
    parser.add_argument("input_file", type=str, help="The input file to parse")
    parser.add_argument(
        "output_file", type=str, help="The output file to write the parsed topics to"
    )
    args = parser.parse_args()
    main(args)
