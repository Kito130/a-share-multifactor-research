# Contributing

Please keep contributions within the public V1 software surface. Do not add
licensed raw or processed market data, credentials, private reports, local
paths, or new claims about the frozen OOS result.

Run the public synthetic tests before opening a pull request:

```powershell
python -m pytest -q
```

The formal P1-P6 artifact tests require a separate private workspace and are not
part of the public reproducibility contract.

AI-assisted development tools were used for implementation support and
documentation review. The research questions, data boundaries, methodology,
verification procedures, and final claims were selected and validated by the
author.
