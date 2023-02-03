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

### Robust04

```
map                     all     0.2840
P_20                    all     0.4060
ndcg_cut_20             all     0.4760
```
