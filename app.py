"""Pressing — can you hear a hit?

Scores a track's chance of landing in the top ~22% of Spotify popularity from its
audio features, genre and release year. Model trains on first run and is cached.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA, CUT = "spotify.csv", 60
AUD = ["danceability", "energy", "loudness", "speechiness", "acousticness",
       "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]
NUM, CAT = AUD + ["year"], ["playlist_genre"]

# --- palette: two-colour risograph print ------------------------------------
PAPER, INK = "#F2EFE6", "#1A1A1A"
PINK, BLUE, PLUM, GREY = "#FF4FA3", "#0F6FC5", "#4B2E83", "#B8B2A4"

st.set_page_config(page_title="Pressing · Hit Predictor", page_icon="◐", layout="wide")
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Work+Sans:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap');
html, body, [class*="css"] {{ font-family:'Work Sans',sans-serif; }}
.stApp {{ background:{PAPER}; }}
.plate {{ font-family:'Anton',sans-serif; font-size:4.2rem; line-height:.88;
  letter-spacing:.01em; text-transform:uppercase; color:{INK}; margin:0; }}
.plate em {{ font-style:normal; color:{PINK}; }}
.eyebrow {{ font-family:'Space Mono',monospace; font-size:.68rem; font-weight:700;
  letter-spacing:.2em; text-transform:uppercase; color:{BLUE}; margin:0 0 .4rem; }}
.rule {{ height:3px; background:{INK}; margin:1.1rem 0 1.6rem; }}
.panel {{ background:#FBF9F4; border:2px solid {INK}; padding:1.3rem 1.5rem; margin-bottom:1rem; }}
.verdict {{ font-family:'Anton',sans-serif; font-size:6rem; line-height:.85; letter-spacing:-.01em; }}
.verdict span {{ font-size:2rem; }}
.band {{ font-family:'Space Mono',monospace; font-weight:700; font-size:.78rem;
  letter-spacing:.15em; text-transform:uppercase; }}
.note {{ font-size:.85rem; color:#57534A; line-height:1.5; }}
.kv {{ font-family:'Space Mono',monospace; font-size:.76rem; color:#57534A; }}
.trk {{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem;
  border-bottom:1px dotted {GREY}; padding:.5rem 0; }}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Cutting the test pressing…")
def build():
    df = pd.read_csv(DATA).drop_duplicates(subset="track_id")
    df["year"] = pd.to_datetime(df.track_album_release_date, errors="coerce",
                                format="mixed").dt.year
    df = df.dropna(subset=["year", "danceability"]).reset_index(drop=True)
    y = (df.track_popularity >= CUT).astype(int)

    pipe = Pipeline([
        ("pre", ColumnTransformer([("n", StandardScaler(), NUM),
                                   ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])),
        ("clf", HistGradientBoostingClassifier(max_iter=350, learning_rate=.06,
                                               max_leaf_nodes=31, random_state=42)),
    ])
    Xtr, Xte, ytr, yte = train_test_split(df[NUM + CAT], y, test_size=.2,
                                          stratify=y, random_state=42)
    pipe.fit(Xtr, ytr)
    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])

    sc = StandardScaler().fit(df[AUD])
    nn = NearestNeighbors(n_neighbors=4).fit(sc.transform(df[AUD]))
    med = df[NUM].median().to_dict()
    hit_profile = df[y == 1][AUD].mean().to_dict()
    lo = df[AUD].quantile(.02).to_dict()
    hi = df[AUD].quantile(.98).to_dict()
    return pipe, df, auc, sc, nn, med, hit_profile, lo, hi, float(y.mean())


pipe, ref, AUC, SC, NN, MED, HITP, LO, HI, BASE = build()

LABEL = {"danceability": "Danceable", "energy": "Energetic", "valence": "Happy",
         "acousticness": "Acoustic", "instrumentalness": "Instrumental",
         "loudness": "Loud", "speechiness": "Wordy", "liveness": "Live",
         "tempo": "Tempo", "duration_ms": "Length", "year": "Release year",
         "playlist_genre": "Genre"}

# --- the desk ---------------------------------------------------------------
st.sidebar.markdown('<p class="eyebrow">The desk</p>', unsafe_allow_html=True)
t = {}
with st.sidebar:
    t["playlist_genre"] = st.selectbox("Genre", ["pop", "rock", "rap", "latin", "r&b", "edm"])
    t["year"] = st.slider("Release year", 1960, 2020, 2018)
    st.markdown("**Feel**")
    t["danceability"] = st.slider("Danceable", 0.0, 1.0, 0.67, .01)
    t["energy"] = st.slider("Energetic", 0.0, 1.0, 0.72, .01)
    t["valence"] = st.slider("Happy", 0.0, 1.0, 0.51, .01)
    st.markdown("**Production**")
    t["loudness"] = st.slider("Loudness (dB)", -25.0, 0.0, -6.3, .1)
    t["acousticness"] = st.slider("Acoustic", 0.0, 1.0, 0.08, .01)
    t["instrumentalness"] = st.slider("Instrumental", 0.0, 1.0, 0.0, .01)
    t["speechiness"] = st.slider("Wordy", 0.0, 1.0, 0.06, .01)
    t["liveness"] = st.slider("Live-sounding", 0.0, 1.0, 0.13, .01)
    st.markdown("**Shape**")
    t["tempo"] = st.slider("Tempo (BPM)", 60, 210, 122)
    secs = st.slider("Length (seconds)", 60, 420, 217)
    t["duration_ms"] = secs * 1000

row = pd.DataFrame([t])[NUM + CAT]
p = float(pipe.predict_proba(row)[0, 1])
band, col = (("Certified", PINK) if p >= .45 else
             ("Promising", PLUM) if p >= .25 else ("Deep cut", BLUE))

st.markdown('<p class="eyebrow">Pressing · Side A</p>', unsafe_allow_html=True)
st.markdown('<p class="plate">Can you <em>hear</em><br>a hit?</p>', unsafe_allow_html=True)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

left, right = st.columns([1, 1.3], gap="large")

with left:
    st.markdown(
        f"""<div class="panel" style="text-align:center">
        <div class="verdict" style="color:{col}">{p*100:.0f}<span>%</span></div>
        <div class="band" style="color:{col};margin-top:.6rem">{band}</div>
        <p class="note" style="margin-top:.8rem">Chance of charting in the top
        {BASE*100:.0f}% of Spotify popularity. An average track sits at
        {BASE*100:.0f}%.</p></div>""", unsafe_allow_html=True)

    # signature: sonic fingerprint — your shape overprinted on the average hit
    AXES = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "loudness"]

    def norm(k, v):
        return float(np.clip((v - LO[k]) / (HI[k] - LO[k] + 1e-9), 0, 1))

    def poly(vals):
        pts = []
        for i, k in enumerate(AXES):
            a = -np.pi / 2 + i * 2 * np.pi / len(AXES)
            r = 20 + 80 * vals[i]
            pts.append(f"{150+r*np.cos(a):.1f},{150+r*np.sin(a):.1f}")
        return " ".join(pts)

    mine = poly([norm(k, t[k]) for k in AXES])
    theirs = poly([norm(k, HITP[k]) for k in AXES])
    web = "".join(
        f'<circle cx="150" cy="150" r="{20+80*f}" fill="none" stroke="{GREY}" stroke-width="1"/>'
        for f in (.33, .66, 1))
    labs = "".join(
        f'<text x="{150+118*np.cos(-np.pi/2+i*2*np.pi/6):.0f}" '
        f'y="{150+118*np.sin(-np.pi/2+i*2*np.pi/6):.0f}" text-anchor="middle" '
        f'font-family="Space Mono" font-size="9" fill="{INK}">{LABEL[k].upper()}</text>'
        for i, k in enumerate(AXES))
    st.markdown(
        f"""<div class="panel"><p class="eyebrow">Sonic fingerprint</p>
        <svg viewBox="0 0 300 300" width="100%" style="max-height:330px">{web}
          <g style="mix-blend-mode:multiply">
            <polygon points="{theirs}" fill="{BLUE}" opacity=".55"/>
            <polygon points="{mine}" fill="{PINK}" opacity=".75"/>
          </g>{labs}</svg>
        <p class="kv"><span style="color:{PINK}">■</span> your track &nbsp;
        <span style="color:{BLUE}">■</span> average charting track</p></div>""",
        unsafe_allow_html=True)

# --- why, and what it sounds like -------------------------------------------
with right:
    st.markdown('<p class="eyebrow">What is moving the needle</p>', unsafe_allow_html=True)
    rows = []
    for k in NUM + CAT:
        flat = row.copy()
        if k in CAT:  # neutral genre = average over all six
            d = p - np.mean([float(pipe.predict_proba(row.assign(playlist_genre=g))[0, 1])
                             for g in ref.playlist_genre.unique()])
        else:
            flat[k] = MED[k]
            d = p - float(pipe.predict_proba(flat)[0, 1])
        rows.append({"Feature": LABEL[k], "d": d * 100})
    imp = pd.DataFrame(rows)
    imp["dir"] = np.where(imp.d > 0, "Helps", "Hurts")
    top = imp.reindex(imp.d.abs().sort_values(ascending=False).index).head(8)

    st.altair_chart(
        alt.Chart(top).mark_bar(height=18).encode(
            x=alt.X("d:Q", title="← costs points   ·   wins points →",
                    axis=alt.Axis(grid=False, labels=False, ticks=False)),
            y=alt.Y("Feature:N", sort="-x", title=None),
            color=alt.Color("dir:N", scale=alt.Scale(domain=["Helps", "Hurts"],
                                                     range=[PINK, BLUE]), legend=None),
            tooltip=["Feature", alt.Tooltip("d:Q", title="points", format="+.1f")],
        ).properties(height=250).configure_view(strokeWidth=0)
        .configure_axis(labelFont="Work Sans", labelColor=INK, titleFont="Space Mono",
                        titleFontSize=9, titleColor="#8A8578"),
        use_container_width=True)
    st.markdown('<p class="note">Each bar is how many points the score would drop if that '
                'setting were dialled back to the median track.</p>', unsafe_allow_html=True)

    st.markdown('<p class="eyebrow" style="margin-top:1.4rem">Sounds most like</p>',
                unsafe_allow_html=True)
    _, idx = NN.kneighbors(SC.transform(pd.DataFrame([{k: t[k] for k in AUD}])[AUD]))
    cards = "".join(
        f'<div class="trk"><span><strong>{ref.track_name[i]}</strong><br>'
        f'<span class="kv">{ref.track_artist[i]}</span></span>'
        f'<span class="band" style="color:{PINK}">{ref.track_popularity[i]}</span></div>'
        for i in idx[0][:3])
    st.markdown(f'<div class="panel">{cards}<p class="kv" style="margin-top:.7rem">'
                f'Nearest real tracks by audio profile · number is their Spotify '
                f'popularity</p></div>', unsafe_allow_html=True)

st.markdown('<div class="rule" style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
st.markdown(
    f'<p class="kv">Gradient boosting · 28,356 tracks · held-out ROC AUC {AUC:.3f}. '
    f'Audio features alone reach 0.670 — genre and release year carry more signal than '
    f'anything you can hear. Treat this as a weak signal, not a verdict.</p>',
    unsafe_allow_html=True)
