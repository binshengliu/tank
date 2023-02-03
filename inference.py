#!/usr/bin/env python
import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple

import more_itertools as mi
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

color_fmt = "[%(asctime)s][%(name)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"
log_fmt = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
logger = logging.getLogger(__file__)


def configure_logging(log_level: str = "INFO", path: Optional[str] = None) -> None:
    level = logging.getLevelName(log_level)
    try:
        import colorlog

        colorlog.basicConfig(level=level, format=color_fmt)
    except Exception:
        logging.basicConfig(level=level, format=log_fmt)

    if path is not None:
        fh = logging.FileHandler(os.path.expanduser(path))
        fh.setFormatter(logging.Formatter(log_fmt))
        logging.root.addHandler(fh)


def split_to_passages(
    query: str, doc: str, window: int, step: int
) -> Tuple[List[str], List[str]]:
    windows = mi.windowed(doc.split(), n=window, step=step)
    parts = [[word for word in x if word is not None] for x in windows]
    psgs = [" ".join(x) for x in parts]
    queries = [query] * len(psgs)
    return queries, psgs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", default=sys.stdin, type=argparse.FileType("r"))
    parser.add_argument("--output", default=sys.stdout, type=argparse.FileType("w"))
    parser.add_argument("--model", default="bsl/bart-ranker")
    parser.add_argument("--sep", default="\t")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--step", type=int, default=200)

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_arguments()

    logger.info(f"Load model {args.model}")
    tokenizer = AutoTokenizer.from_pretrained("bsl/bart-ranker")
    model = AutoModelForSequenceClassification.from_pretrained("bsl/bart-ranker")
    model.eval()

    logger.info("Ready")
    for line in args.input:
        cols = line.split(args.sep)
        if len(cols) != 2:
            logger.warning("Please separate the query and document by tab.")
            continue
        query, doc = cols
        queries, psgs = split_to_passages(query, doc, args.window, args.step)

        inputs = tokenizer(
            queries,
            psgs,
            return_tensors="pt",
            padding=True,
            truncation="only_second",
            max_length=args.max_length,
        )
        with torch.no_grad():
            score = model(**inputs).logits.max().item()
        args.output.write(f"{score:f}\n")


if __name__ == "__main__":
    main()
