# Vendored AASIST

Source repository: https://github.com/clovaai/aasist
Ref: `main`  |  Commit at vendoring time: `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`
Vendored on (UTC): 2026-09-02T08:26:06+00:00
License: MIT (NAVER Corp.) -- see `LICENSE` and `NOTICE` in this directory.

## Files

| Local file | Upstream path | sha256 |
| --- | --- | --- |
| `aasist_model.py` | `models/AASIST.py` | `9e0d3e80937dd0577beea7883098465a479da23a198ebc0d712abcc59b0bec50` |
| `AASIST.pth` | `models/weights/AASIST.pth` | `51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0` |
| `LICENSE` | `LICENSE` | `da2e79b8592d166ef505224300968b80ebe1e4c217c43b94a5ec627d81cd4142` |
| `NOTICE` | `NOTICE` | `70edb07f04ddc88e7155d6e826495e5ef99c3b1f0bd4e1b7fa35897b3854dbd0` |

## FULL AASIST confirmation

`weights/AASIST.pth` is **1281532 bytes** -- this is the FULL AASIST
checkpoint. The light variant `AASIST-L.pth` is 426428 bytes and is
deliberately NOT used by VenoDec. `download_aasist.py` refuses to write the file
if the size does not match, and the model is loaded with `strict=True` against the
FULL AASIST architecture config, which would raise on AASIST-L weights.

`aasist_model.py` is an unmodified copy of upstream `models/AASIST.py`.
