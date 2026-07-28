# Can You Hear a Hit?

Built by **Megha Mishra**.

Set a track using the controls — genre, year, and ten audio features — and the app
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

## Audio playback

The "Sounds most like" panel embeds Spotify's own player for each matched track
using the `track_id` already in the dataset — no API key and no credentials
needed. Without a Spotify login you get a 30-second preview; logged in, the full
track plays. A few tracks are unavailable in some countries and will show a
greyed-out player.

Note that this is the *only* audio in the app. The predictor works from numeric
descriptions of sound, not from sound itself.

## Snapshot limitation

The latest release date anywhere in the file is **29 January 2020**, and all 626
tracks dated 2020 are from that single month. The dataset is a one-off scrape, not
a live feed, so nothing from the last six years appears in it.

This matters for interpreting the model. 2019 alone contributes 7,406 tracks —
more than double any other year — and Spotify popularity weights recent plays. A
2019 song was at peak rotation exactly when the snapshot was taken. So release
year, the model's second-strongest feature, is partly measuring *when the data was
collected* rather than what makes a song succeed.

Extending the data is not straightforward either: Spotify deprecated the
`audio-features` endpoint for new applications in late 2024, so the ten audio
columns this app depends on can no longer be fetched with fresh credentials.

## Coverage limitation

The training data comes from Western Spotify playlists across six genres: pop,
rock, rap, latin, r&b and edm. Counting artist names, **only 5 of the 28,356
tracks are by Indian artists** — two by Badshah, one each by Neha Kakkar, Diljit
Dosanjh and Anirudh Ravichander.

The model therefore has no meaningful exposure to Bollywood, Punjabi, Tamil or
Indian classical music, and its predictions for them are not trustworthy. This is
stated in the app footer rather than left for a user to discover.

Closing the gap properly would mean more than appending rows. Spotify popularity
is a global measure, so Indian tracks score lower for reasons of audience size
rather than musical quality — a naive merge would teach the model that Indian
music does not chart, which is an artefact of the metric, not a finding about the
music. It would need a language feature, per-market popularity normalisation, and
a fresh evaluation.

## Two things to know before you present this

**Survivorship bias.** Set the year to 1965 and the score *rises*. Old tracks only
appear in these playlists if they became classics, so the model learns "old =
good." It is a real pattern in the data and a fake pattern in the world. Good
material for a viva question.

**Popularity is not quality.** Spotify's popularity score reflects recent play
counts, so it partly measures promotion budget and playlist placement. The model
is predicting commercial traction, not whether a song is good.

## Design notes

Restrained, document-like layout: a bone paper background, a titling block with a
hairline rule, and light panel borders. Work Sans for headings and body, Space
Mono for figures and labels. Colour is used only where it carries meaning —
fluorescent pink and process blue appear in the chart bars and the fingerprint,
not as decoration.

The signature element is the **sonic fingerprint**: a six-axis radar where the
invented track prints over the average charting track using
`mix-blend-mode: multiply`, so the two shapes overprint where they overlap. Where
they disagree is where the track differs from what charts.

## Extending it

- **Regression mode** — predict the 0–100 score instead of a yes/no. Lower R², but
  a more granular readout.
- **Drop year and genre** — rebuild on audio only and show how much worse it gets.
  That comparison is arguably a more interesting project than the predictor.
- **Per-genre models** — the genre effect is so strong that six small models may
  beat one big one.

## Licence

© 2026 Megha Mishra. All rights reserved.

The application code and design are mine. The Spotify songs dataset is from
TidyTuesday and remains the property of its original publishers; audio previews
are served by Apple Music and belong to their respective rights holders.
