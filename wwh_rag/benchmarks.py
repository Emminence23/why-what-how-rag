"""Load the eight benchmarks as one common multiple-choice format."""
import random
from dataclasses import dataclass, field

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

AVERITEC_LABELS = ["Supported", "Refuted", "Not Enough Evidence",
                   "Conflicting Evidence/Cherrypicking"]


@dataclass
class Item:
    id: str
    question: str
    options: list
    gold: int                      # index into options
    context: str = ""              # MuSR narrative / AveriTeC claim
    meta: dict = field(default_factory=dict)

    @property
    def gold_letter(self):
        return LETTERS[self.gold]


# What the model is told the questions are about (used inside prompts).
DOMAINS = {
    "arc": "grade-school science",
    "hellaswag": "commonsense story continuation",
    "gpqa": "graduate-level science",
    "mmlu": "general knowledge across many subjects",
    "medmcqa": "medical entrance exam",
    "musr": "narrative reasoning (read the story carefully)",
    "truthfulqa": "common misconceptions (prefer the literally true answer)",
    "averitec": "fact-checking real-world claims",
}


def _shuffled(options, gold, rng):
    order = list(range(len(options)))
    rng.shuffle(order)
    return [options[i] for i in order], order.index(gold)


def _load(name):
    from datasets import load_dataset

    if name == "arc":
        for i, r in enumerate(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")):
            labels = r["choices"]["label"]
            yield Item(r["id"], r["question"], r["choices"]["text"], labels.index(r["answerKey"]))
    elif name == "hellaswag":
        for r in load_dataset("Rowan/hellaswag", split="validation"):
            yield Item(str(r["ind"]), r["ctx"], [e.strip() for e in r["endings"]], int(r["label"]),
                       meta={"activity": r["activity_label"]})
    elif name == "gpqa":  # gated: accept the terms on Hugging Face and set HF_TOKEN
        for i, r in enumerate(load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")):
            opts = [r["Correct Answer"], r["Incorrect Answer 1"],
                    r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            opts, gold = _shuffled([o.strip() for o in opts], 0, random.Random(i))
            yield Item(str(i), r["Question"].strip(), opts, gold)
    elif name == "mmlu":
        for i, r in enumerate(load_dataset("cais/mmlu", "all", split="test")):
            yield Item(str(i), r["question"], r["choices"], int(r["answer"]),
                       meta={"subject": r["subject"]})
    elif name == "medmcqa":
        for r in load_dataset("openlifescienceai/medmcqa", split="validation"):
            yield Item(r["id"], r["question"], [r["opa"], r["opb"], r["opc"], r["opd"]], int(r["cop"]))
    elif name == "musr":
        import ast
        ds = load_dataset("TAUR-Lab/MuSR")
        for split in ["murder_mysteries", "object_placements", "team_allocation"]:
            for i, r in enumerate(ds[split]):
                choices = r["choices"]
                if isinstance(choices, str):
                    choices = ast.literal_eval(choices)
                yield Item(f"{split}/{i}", r["question"], list(choices), int(r["answer_index"]),
                           context=r["narrative"], meta={"split": split})
    elif name == "truthfulqa":  # MC1: exactly one true option, listed first in the data
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
        for i, r in enumerate(ds):
            opts, gold = _shuffled(r["mc1_targets"]["choices"], 0, random.Random(i))
            yield Item(str(i), r["question"], opts, gold)
    elif name == "averitec":  # pick one of four verdicts for the claim
        for r in load_dataset("AlbertHatsuki/AveriTeC-decomposed", split="train"):
            yield Item(str(r["id"]), "Which verdict best describes this claim?",
                       AVERITEC_LABELS, AVERITEC_LABELS.index(r["label"]), context=r["claim"])
    else:
        raise ValueError(f"unknown benchmark {name!r}; choose from {sorted(DOMAINS)}")


def load_benchmark(name, n=100, seed=42):
    """A fixed random sample of n items (same seed -> same questions)."""
    items = list(_load(name))
    random.Random(seed).shuffle(items)
    return items[:n]
