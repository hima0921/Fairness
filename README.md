# Fairness

## How to load the data

```python
from src.data.bbq_loader import load_bbq
table = load_bbq()
train = table[table["split"] == "train"]
```

Note: run this from the repo's top folder (the one with this README), or the `src` import won't be found.

`load_bbq()` downloads BBQ from GitHub, so it needs an internet connection. See [docs/schema.md](docs/schema.md) for what the columns mean and [docs/splits.md](docs/splits.md) for how the splits work.
