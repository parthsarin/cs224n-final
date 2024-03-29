"""
File: beam_search.py
--------------------

Runs beam search with a "teacher" model to generate prompts for the "student" model,
which is imported from annotate.py.
"""
import argparse
from openai import OpenAI
import dataclasses
from dataclasses import dataclass
from annotate import OpenAIClassifier, HuggingFaceClassifier, ArticleClassifier
from glob import glob
import json
from json import load, dump
from dataclasses_json import dataclass_json
from tqdm import tqdm


# ------------------------------------------------------------------------------
# score utilities
# ------------------------------------------------------------------------------
@dataclass_json
@dataclass
class Prompt:
    text: str = ""
    score: float = -1
    generation: int = 0


class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder taken from https://stackoverflow.com/questions/51286748/make-the-python-json-encoder-support-pythons-new-dataclasses"""

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def is_correct(label, pred):
    """
    checks if the model's label for the file was correct.

    label, pred -- should be formatted as {funding source: probability}
    """
    try:
        return all(label[k] == round(pred[k]) for k in label)
    except Exception:
        return False


def score(labels, preds):
    """
    returns the model's accuracy on all of the files in labels.

    labels, preds -- should be formatted as {filename: {funding source: probability}}
    """
    return sum(is_correct(labels[k], preds[k]) for k in labels) / len(labels)


def eval_generation(args, student: ArticleClassifier, generation):
    print("beginning evaluation")
    articles = {
        filename: load(open(filename, "r"))
        for filename in glob(f"{args.in_dir}/*.json")
    }
    labels = {filename: article["funding"] for filename, article in articles.items()}
    with tqdm(total=len(generation) * len(articles)) as pbar:
        for p_idx, prompt in enumerate(generation):
            # set the pbar prompt info to the prompt idx
            pbar.set_postfix_str(f"prompt {p_idx + 1} / {len(generation)}")
            preds = {}
            for filename, article in articles.items():
                try:
                    preds[filename] = student.annotate(article["article"], prompt.text)
                except Exception as e:
                    print(f"error on {filename}: {e}")
                    preds[filename] = None
                pbar.update(1)
            prompt.score = score(labels, preds)


# ------------------------------------------------------------------------------
# models
# ------------------------------------------------------------------------------
class OpenAITeacher:
    def __init__(self, model, prompt_file="prompts/teacher.txt", temperature=1):
        self.client = OpenAI()
        self.model = model
        self.prompt_template = open(prompt_file, "r").read()
        self.temperature = temperature

    def perturb(self, prompt: Prompt) -> Prompt:
        # fill out the prompt and send it to the model
        prompt_text = prompt.text.format(article=r"{article}")
        messages = [
            {
                "role": "user",
                "content": self.prompt_template.format(
                    accuracy=prompt.score, prompt=prompt_text
                ),
            }
        ]
        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            n=1,
        )
        r = str(r.choices[0].message.content)

        # get a new prompt with the new text
        return Prompt(r)

    def __repr__(self):
        return f'OpenAITeacher(model="{self.model}")'


def print_generation(gen_idx, generation):
    print(f"generation {gen_idx}:")
    print(f"- best score: {generation[0].score}")
    print(f"- best prompt: {generation[0].text}")
    print(f"- worst score: {generation[-1].score}")
    print()


def save_prompts(args, prompts):
    with open(args.out_file, "w") as f:
        dump(prompts, f, cls=EnhancedJSONEncoder, indent=4)


def main(args):
    # set up the student model
    print(
        f"setting up student model {args.student_model} with temperature {args.student_temperature}"
    )
    if args.student_model.startswith("gpt"):
        student = OpenAIClassifier(
            args.student_model,
            temperature=args.student_temperature,
        )
        generation = [Prompt(open("prompts/annotate-openai.txt").read())]
    else:
        student = HuggingFaceClassifier(
            args.student_model,
            temperature=args.student_temperature,
        )
        generation = [Prompt(open("prompts/annotate-hf.txt").read())]

    # set up the teacher model
    print(
        f"setting up teacher model {args.teacher_model} with temperature {args.teacher_temperature}"
    )
    teacher = OpenAITeacher(
        args.teacher_model,
        temperature=args.teacher_temperature,
    )

    # load the last generation
    try:
        all_prompts = load(open(args.out_file, "r"))
        all_prompts = [Prompt.from_dict(p) for p in all_prompts]
    except FileNotFoundError:
        all_prompts = []

        eval_generation(args, student, generation)
        all_prompts += generation
        save_prompts(args, all_prompts)
        print_generation(0, generation)
    else:
        gen_idx = all_prompts[-1].generation
        print(f"loaded prompts from file, current generation is {gen_idx}")
        generation = [p for p in all_prompts if p.generation == gen_idx]
        eval_generation(args, student, generation)
        generation.sort(key=lambda p: p.score, reverse=True)
        save_prompts(args, all_prompts)
        print_generation(0, generation)

    for i in range(1, args.beam_depth + 1):
        # perturb the best prompts
        print("perturbing the best prompts")
        generation = [
            p for p in generation[: args.beam_width] for _ in range(args.beam_degree)
        ]
        generation = [teacher.perturb(p) for p in tqdm(generation)]

        # update and save the generation
        for p in generation:
            p.generation = i
        all_prompts += generation
        save_prompts(args, all_prompts)

        # evaluate the new generation
        eval_generation(args, student, generation)
        generation.sort(key=lambda p: p.score, reverse=True)
        all_prompts = all_prompts[: -len(generation)] + generation
        save_prompts(args, all_prompts)
        print_generation(i, generation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="runs beam search with a teacher model to generate prompts for the student model"
    )

    parser.add_argument(
        "in_dir", type=str, help="input directory containing JSON files"
    )
    parser.add_argument("out_file", type=str, help="output file to save the prompts to")
    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="the number of prompts to keep in each generation",
    )
    parser.add_argument(
        "--beam-degree",
        type=int,
        default=3,
        help="the number of perturbations to generate for each prompt",
    )
    parser.add_argument(
        "--beam-depth", type=int, default=3, help="the number of generations to run"
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="gpt-4-1106-preview",
        help="the AI model to use -- gpt-* will be passed to OpenAI, otherwise HuggingFace",
    )
    parser.add_argument(
        "--teacher-temperature",
        type=float,
        default=1,
        help="temperature for teacher generation",
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default="gpt-4-1106-preview",
        help="the AI model to use -- gpt-* will be passed to OpenAI, otherwise HuggingFace",
    )
    parser.add_argument(
        "--student-temperature",
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

    args = parser.parse_args()
    main(args)
