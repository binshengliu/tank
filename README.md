# Tank (Search Ranking Model)

## Dependencies

```
pip install transformers
```

## Quick start

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("bsl/bart-ranker")
model = AutoModelForSequenceClassification.from_pretrained("bsl/bart-ranker")
query = "information retrieval"
doc = "Information retrieval (IR) in computing and information science is the process of obtaining information system resources that are relevant to an information"
inputs = tokenizer(query, doc, return_tensors="pt")
score = model(**inputs).logits.item()
```

## Inference

Run a script to accept query-document pairs from stdin and output scores to
stdout. A query-document should be separated by tab "\t".

Long documents can be broken into passages via `--window` and `--step`
parameters. The final score is the max score of all passages.

```
$ python3 inference.py
information retrieval   Information retrieval (IR) in computing and information science is the process of obtaining information system resources that are relevant to an information
5.778058
```

## Effectiveness

|                  | NDCG@10 | NDCG@20 | AP     | P@10   | P@20   | MRR@10 |
|------------------|---------|---------|--------|--------|--------|--------|
| Robust04         | 0.5113  | 0.4760  | 0.2840 | 0.4936 | 0.4060 |        |
| [TREC DL 20][1]  | 0.7536  | 0.7209  | 0.5283 | 0.8093 | 0.6815 |        |
| [MSMARCO Dev][2] |         |         |        |        |        | 0.394  |

### Notes

- Robust04 first-stage: Indri Query Likelihood
- Robust04 document aggregation: MaxP
- TREC DL 20 first-stage: BM25 on [DeepCT][3] enriched collection
- MSMARCO Dev first-stage: BM25 on [DeepCT][3] enriched collection

[1]: https://trec.nist.gov/pubs/trec29/papers/RMIT.DL.pdf
[2]: https://microsoft.github.io/msmarco
[3]: https://github.com/AdeDZY/DeepCT
