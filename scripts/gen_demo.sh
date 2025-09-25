
---

### `scripts/gen_demo.sh`
```bash
#!/usr/bin/env bash
set -e
python -m ardt4d.synth --out ./data/synth --n_scenes 12 --A 22 --R 100 --D 64 --T 10
python -m ardt4d.datasets.build --data ./data/synth --out ./data/ardt4d --split 0.7 0.2 0.1

