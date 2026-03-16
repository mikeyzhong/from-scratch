import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from PIL import Image
    import random

    return Image, Path, mo, pd, plt, random


@app.cell
def _(Path, pd):
    DATA_DIR = Path("data")
    captions_df = pd.read_csv(DATA_DIR / "captions.txt")
    return DATA_DIR, captions_df


@app.cell
def _(captions_df, plt):
    captions_df["word_count"] = captions_df["caption"].str.split().str.len()

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.hist(captions_df["word_count"], bins=range(1, captions_df["word_count"].max() + 2), edgecolor="black", alpha=0.7)
    _ax.set_xlabel("Caption length (words)")
    _ax.set_ylabel("Count")
    _ax.set_title("Caption Length Distribution")
    _ax.axvline(captions_df["word_count"].median(), color="red", linestyle="--", label=f"median = {captions_df['word_count'].median():.0f}")
    _ax.legend()
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(DATA_DIR, Image, pd, plt):
    image_files = list((DATA_DIR / "Images").glob("*.jpg"))
    sizes = []
    for _img_path in image_files:
        with Image.open(_img_path) as _img:
            sizes.append({"width": _img.width, "height": _img.height})
    sizes_df = pd.DataFrame(sizes)

    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.scatter(sizes_df["width"], sizes_df["height"], alpha=0.15, s=4)
    _ax.set_xlabel("Width (px)")
    _ax.set_ylabel("Height (px)")
    _ax.set_title("Image Resolution Scatter")
    _ax.set_aspect("equal")
    plt.tight_layout()
    plt.gca()
    return (image_files,)


@app.cell
def _(Image, captions_df, image_files, plt, random):
    _k = min(9, len(image_files))
    _sample = random.sample(image_files, _k)
    _caption_lookup = captions_df.groupby("image")["caption"].first()

    _fig, _axes = plt.subplots(3, 3, figsize=(10, 10))
    for _ax, img_path in zip(_axes.flat, _sample):
        with Image.open(img_path) as _img:
            _ax.imshow(_img)
        caption = _caption_lookup.get(img_path.name, "")
        _ax.set_title(caption[:50] + "..." if len(caption) > 50 else caption, fontsize=8)
        _ax.axis("off")
    for _ax in _axes.flat[_k:]:
        _ax.axis("off")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(captions_df, mo):
    _punct = captions_df["caption"].str.extractall(r'([.!?,;:"\'\-])')[0].value_counts()
    mo.vstack([
        mo.md("## Punctuation Counts"),
        mo.ui.table(_punct.reset_index(name="character")),
    ])
    return


if __name__ == "__main__":
    app.run()
