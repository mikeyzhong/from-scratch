# CLIP From Scratch

A small image-text retrieval project that trains a CLIP-style model from scratch
on a local captioned image dataset.

## Dataset Layout

The training code expects the Flickr8 dataset.

```text
data/
  captions.txt
  Images/
    *.jpg
```

`captions.txt` should include at least:

- `image`
- `caption`

## Training

Single run:

```bash
uv run python train.py
```

## Search

`search.py` expects a local trained checkpoint named `clip_model.pt`, which is
intentionally excluded from version control.

```bash
uv run python search.py
```
