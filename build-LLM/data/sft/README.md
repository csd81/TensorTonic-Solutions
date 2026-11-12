# Instruction data

`alpaca_subset.json` contains the first 2,000 usable examples from Stanford
Alpaca:

https://github.com/tatsu-lab/stanford_alpaca

The Alpaca data is released under CC BY-NC 4.0 and is intended for research
use. Each record has `instruction`, `input`, and `output` fields. The SFT
script formats them as:

```
### Instruction:
<instruction>
### Input:
<input>            (omitted when empty)
### Response:
<output>
```

Only the response tokens are trained on, using the copied
`cs336-l15-assistant-only-sft-mask` solution.
