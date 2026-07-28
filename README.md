# Pressing — Can You Hear a Hit?

Set a track on the mixing desk — genre, year, and ten audio features — and the app
scores its chance of landing in the top 22% of Spotify popularity, shows which
settings are moving the number, and names the real songs that sound closest.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

Push this folder to a public GitHub repo (keep `spotify.csv` — the app trains on
startup, so there is no model file to version). Then share.streamlit.io →
**New app** → pick the repo → main file `app.py`. First load takes ~15 seconds
while the model trains; after that it is cached.

## Data

TidyTuesday's Spotify songs set, 32,833 rows deduplicated to **28,356 unique
tracks** spanning 1957–2020. A track counts as a hit if its Spotify popularity is
60 or above — the top 22.6%.

## The model

Gradient boosting on 12 features. Held-out **ROC AUC 0.740**.

The honest headline is in the footer of the app: **audio features alone only
reach 0.670.** Genre and release year carry more signal than anything you can
actually hear. Permutation importance, in order:

| Feature | Δ AUC |
|---|---|
| Genre | +0.087 |
| Release year | +0.066 |
| Loudness | +0.051 |
| Instrumentalness | +0.038 |
| Energy | +0.027 |
| everything else | < 0.01 each |

Genre matters enormously on its own: pop tracks hit 35% of the time, EDM only 7%.

Explanations use ablation rather than SHAP — each bar is the drop in score if that
one setting were dialled back to the median track. No extra dependency, and it
reads in the same units as the prediction.

## Two things to know before you present this

**Survivorship bias.** Set the year to 1965 and the score *rises*. Old tracks only
appear in these playlists if they became classics, so the model learns "old =
good." It is a real pattern in the data and a fake pattern in the world. Good
material for a viva question.

**Popularity is not quality.** Spotify's popularity score reflects recent play
counts, so it partly measures promotion budget and playlist placement. The model
is predicting commercial traction, not whether a song is good.

## Design notes

Two-colour risograph gig poster: newsprint paper, fluorescent pink and process
blue, overprinting to plum where they cross. Anton for the poster type, Work Sans
for reading, Space Mono for figures.

The signature element is the **sonic fingerprint** — a six-axis radar where your
track's shape is printed in pink over the average charting track's shape in blue,
using `mix-blend-mode: multiply` so the overlap genuinely overprints the way two
riso passes do. Where the shapes disagree is where your track differs from what
charts.

## Extending it

- **Regression mode** — predict the 0–100 score instead of a yes/no. Lower R², but
  a more granular readout.
- **Drop year and genre** — rebuild on audio only and show how much worse it gets.
  That comparison is arguably a more interesting project than the predictor.
- **Per-genre models** — the genre effect is so strong that six small models may
  beat one big one.
