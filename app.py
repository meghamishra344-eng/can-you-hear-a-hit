"""Pressing — can you hear a hit?

Scores a track's chance of landing in the top ~22% of Spotify popularity from its
audio features, genre and release year. Model trains on first run and is cached.
"""

import html
import json
import re
import urllib.parse
import urllib.request

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
DEVELOPER = "Megha Mishra"
EYEBROW = "Spotify hit predictor \u00b7 machine learning project"
YEAR = 2026
# ---------------------------------------------------------------------------

DATA, CUT = "spotify.csv", 60
AUD = ["danceability", "energy", "loudness", "speechiness", "acousticness",
       "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]
NUM, CAT = AUD + ["year"], ["playlist_genre"]

# --- palette: two-colour risograph print ------------------------------------
PAPER, INK = "#F2EFE6", "#1A1A1A"
PINK, BLUE, PLUM, GREY = "#FF4FA3", "#0F6FC5", "#4B2E83", "#B8B2A4"

st.set_page_config(page_title="Spotify Hit Predictor", page_icon="◐", layout="wide")
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Work+Sans:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap');
html, body, [class*="css"] {{ font-family:'Work Sans',sans-serif; }}
.stApp {{ background:{PAPER}; }}
.hero {{ background:{INK}; padding:1.7rem 2rem 1.5rem; margin:0 0 1.2rem;
  border-bottom:6px solid {PINK}; }}
.hero .plate {{ font-family:'Anton',sans-serif; font-size:5rem !important;
  line-height:.92 !important; letter-spacing:.01em; text-transform:uppercase;
  color:{PAPER}; margin:0; }}
.hero .plate em {{ font-style:normal; color:{PINK}; }}
.hero .byline {{ font-family:'Space Mono',monospace; font-size:.72rem !important;
  letter-spacing:.16em; text-transform:uppercase; color:{PAPER}; opacity:.62; margin:.9rem 0 0; }}
.foot {{ font-family:'Space Mono',monospace; font-size:.72rem; color:#6B665C;
  line-height:1.7; margin-top:.4rem; }}

.sub {{ font-size:1.02rem; line-height:1.55; color:#3D3A33; max-width:46rem; margin:.9rem 0 0; }}
.sub strong {{ color:{INK}; }}
.cap {{ font-size:.8rem; color:#6B665C; line-height:1.45; margin:.1rem 0 .8rem; }}
.eyebrow {{ font-family:'Space Mono',monospace; font-size:.68rem; font-weight:700;
  letter-spacing:.2em; text-transform:uppercase; color:{BLUE}; margin:0 0 .4rem; }}
.rule {{ height:3px; background:{INK}; margin:1.1rem 0 1.6rem; }}
.panel {{ background:#FBF9F4; border:2px solid {INK}; padding:1.3rem 1.5rem; margin-bottom:1rem; }}
.verdict {{ font-family:'Anton',sans-serif; font-size:5.4rem; line-height:1.06;
  letter-spacing:-.01em; padding-top:.4rem; }}
.verdict span {{ font-size:2rem; }}
.band {{ font-family:'Space Mono',monospace; font-weight:700; font-size:.78rem;
  letter-spacing:.15em; text-transform:uppercase; }}
.note {{ font-size:.85rem; color:#57534A; line-height:1.5; }}
.kv {{ font-family:'Space Mono',monospace; font-size:.76rem; color:#57534A; }}
.trk {{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem;
  border-bottom:1px dotted {GREY}; padding:.5rem 0; }}
section[data-testid="stSidebar"] {{ background:#E7E1D2; border-right:2px solid {INK}; }}
section[data-testid="stSidebar"] .stSlider label p,
section[data-testid="stSidebar"] .stSelectbox label p,
section[data-testid="stSidebar"] .stRadio label p {{ font-size:.86rem !important;
  font-weight:500; color:#3D3A33; }}
.sect {{ font-family:'Space Mono',monospace; font-size:.66rem; font-weight:700;
  letter-spacing:.19em; text-transform:uppercase; color:{INK}; opacity:.75;
  margin:1.7rem 0 .3rem; padding-top:.85rem; border-top:1px solid #CDC5B2; }}
.panel {{ box-shadow:5px 5px 0 rgba(26,26,26,.07); }}
.block-container {{ padding-top:2.2rem; padding-bottom:3rem; }}
.stSlider [data-baseweb="slider"] {{ margin-bottom:.1rem; }}
hr {{ border-color:#CDC5B2; }}
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
    z = sc.transform(df[AUD]).astype("float32")
    med = df[NUM].median().to_dict()
    hit_profile = df[y == 1][AUD].mean().to_dict()
    lo = df[AUD].quantile(.02).to_dict()
    hi = df[AUD].quantile(.98).to_dict()
    return pipe, df, auc, sc, z, med, hit_profile, lo, hi, float(y.mean())


@st.cache_data(ttl=86400, show_spinner=False)
def preview(name, artist):
    """30-second MP3 from Apple's public search API. No key, no login.

    Titles like "SUPREME - L'ego (feat. X)" rarely match, so we retry with the
    remix/feature clutter stripped before giving up.
    """
    bare = re.sub(r"\s*[\(\[][^)\]]*[\)\]]", "",
                  re.sub(r"\s+[-\u2013]\s+.*$", "", str(name))).strip()
    for term in (f"{artist} {name}", f"{artist} {bare}", bare):
        if not term.strip():
            continue
        q = urllib.parse.urlencode({"term": term, "entity": "song", "limit": 5})
        try:
            with urllib.request.urlopen(f"https://itunes.apple.com/search?{q}", timeout=5) as r:
                for hit in json.load(r).get("results") or []:
                    if hit.get("previewUrl"):
                        return hit["previewUrl"]
        except Exception:
            pass
    return None


pipe, ref, AUC, SC, Z, MED, HITP, LO, HI, BASE = build()

LABEL = {"danceability": "Danceable", "energy": "Energetic", "valence": "Happy",
         "acousticness": "Acoustic", "instrumentalness": "Instrumental",
         "loudness": "Loud", "speechiness": "Wordy", "liveness": "Live",
         "tempo": "Tempo", "duration_ms": "Length", "year": "Release year",
         "playlist_genre": "Genre"}

# --- the desk ---------------------------------------------------------------
st.sidebar.markdown('<p class="eyebrow">The desk</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="cap">You are describing a song that does not exist. '
                    'Each control below is one thing about how it sounds.</p>',
                    unsafe_allow_html=True)
t = {}
with st.sidebar:
    t["playlist_genre"] = st.selectbox("Genre", ["pop", "rock", "rap", "latin", "r&b", "edm"])
    t["year"] = st.slider("Release year", 1960, 2020, 2018)
    st.markdown('<div class="sect">Feel</div>', unsafe_allow_html=True)
    t["danceability"] = st.slider("Danceable", 0.0, 1.0, 0.67, .01)
    t["energy"] = st.slider("Energetic", 0.0, 1.0, 0.72, .01)
    t["valence"] = st.slider("Happy", 0.0, 1.0, 0.51, .01)
    st.markdown('<div class="sect">Production</div>', unsafe_allow_html=True)
    t["loudness"] = st.slider("Loudness (dB)", -25.0, 0.0, -6.3, .1)
    t["acousticness"] = st.slider("Acoustic", 0.0, 1.0, 0.08, .01)
    t["instrumentalness"] = st.slider("Instrumental", 0.0, 1.0, 0.0, .01)
    t["speechiness"] = st.slider("Wordy", 0.0, 1.0, 0.06, .01)
    t["liveness"] = st.slider("Live-sounding", 0.0, 1.0, 0.13, .01)
    st.markdown('<div class="sect">Shape</div>', unsafe_allow_html=True)
    t["tempo"] = st.slider("Tempo (BPM)", 60, 210, 122)
    secs = st.slider("Length (seconds)", 60, 420, 217)
    t["duration_ms"] = secs * 1000

row = pd.DataFrame([t])[NUM + CAT]
p = float(pipe.predict_proba(row)[0, 1])
band, col = (("Likely to chart", PINK) if p >= .45 else
             ("Borderline", PLUM) if p >= .25 else ("Unlikely to chart", BLUE))

st.markdown(
    f'<div class="hero">'
    f'<div class="plate">Can you <em>hear</em><br>a hit?</div>'
    f'<p class="eyebrow" style="color:#7FB2E8;margin:1rem 0 0">{html.escape(EYEBROW)}</p>'
    f'<div class="byline" style="margin:.35rem 0 0">Built and deployed by '
    f'{html.escape(DEVELOPER)}</div>'
    f'</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub"><strong>Invent a song using the controls on the left.</strong> '
    'A model trained on 28,356 real Spotify tracks estimates the chance your invented '
    'song would chart, shows which of your choices moved that number, and plays the '
    'real songs that sound closest to it.</p>', unsafe_allow_html=True)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

left, right = st.columns([1, 1.3], gap="large")

with left:
    st.markdown(
        f"""<div class="panel" style="text-align:center">
        <div class="verdict" style="color:{col}">{p*100:.0f}<span>%</span></div>
        <div class="band" style="color:{col};margin-top:.6rem">{band}</div>
        <p class="note" style="margin-top:.8rem">Odds of landing in the top
        {BASE*100:.0f}% by Spotify popularity. A track picked at random has a
        {BASE*100:.0f}% shot — this one is {'better' if p > BASE else 'worse'}
        than that.</p></div>""", unsafe_allow_html=True)

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
        f"""<div class="panel"><p class="eyebrow">Sonic fingerprint</p><p class="cap">Your invented song, printed over the shape of an average charting track. Where they disagree is where yours differs.</p>
        <svg viewBox="0 0 300 300" width="100%" style="max-height:330px">{web}
          <g style="mix-blend-mode:multiply">
            <polygon points="{theirs}" fill="{BLUE}" fill-opacity=".5" stroke="{BLUE}" stroke-width="2.5"/>
            <polygon points="{mine}" fill="{PINK}" fill-opacity=".5" stroke="{PINK}" stroke-width="2.5"/>
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
    yv, gv = ref.year.values, ref.playlist_genre.values
    y0, g0 = t["year"], t["playlist_genre"]
    for mask, pool in [
        ((np.abs(yv - y0) <= 4) & (gv == g0), f"{g0}, {y0-4}\u2013{y0+4}"),
        ((np.abs(yv - y0) <= 12) & (gv == g0), f"{g0}, {y0-12}\u2013{y0+12}"),
        (np.abs(yv - y0) <= 8, f"any genre, {y0-8}\u2013{y0+8}"),
        (np.ones(len(ref), bool), "the whole catalogue"),
    ]:
        cand = np.flatnonzero(mask)
        if len(cand) >= 6:
            break
    q = SC.transform(pd.DataFrame([{k: t[k] for k in AUD}])[AUD])[0]
    ranked = cand[np.argsort(((Z[cand] - q) ** 2).sum(1))[:6]]
    found = []
    for j in ranked:
        found.append((j, preview(str(ref.track_name[j]), str(ref.track_artist[j]))))
        if sum(1 for _, u in found if u) >= 3:
            break
    picks = ([r for r in found if r[1]] + [r for r in found if not r[1]])[:3]
    for i, url in picks:
        st.markdown(
            f'<div class="trk" style="border-bottom:none;padding-bottom:.15rem">'
            f'<span><strong>{html.escape(str(ref.track_name[i]))}</strong>'
            f'<span class="kv"> · {html.escape(str(ref.track_artist[i]))}</span></span>'
            f'<span class="band" style="color:{PINK}">{ref.track_popularity[i]}</span></div>',
            unsafe_allow_html=True)
        if url:
            st.audio(url)
        else:
            st.markdown(
                f'<a class="kv" href="https://open.spotify.com/track/{ref.track_id[i]}" '
                f'target="_blank">no preview found &mdash; open in Spotify &rarr;</a>',
                unsafe_allow_html=True)
    st.markdown(f'<p class="kv" style="margin-top:.6rem">Closest real tracks drawn from '
                f'<strong>{pool}</strong> ({len(cand):,} songs) · number is their '
                f'Spotify popularity · 30-second previews via Apple Music</p>', unsafe_allow_html=True)

st.markdown('<div class="rule" style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
st.markdown(
    f'<p class="foot">&copy; {YEAR} {html.escape(DEVELOPER)}. All rights reserved.<br>'
    f'Data: TidyTuesday Spotify songs set · previews via Apple Music · '
    f'built with Streamlit and scikit-learn.</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="kv">Gradient boosting · 28,356 tracks · held-out ROC AUC {AUC:.3f}. '
    f'Audio features alone reach 0.670 — genre and release year carry more signal than '
    f'anything you can hear. Treat this as a weak signal, not a verdict.<br><br>'
    f'<strong>Known limitation:</strong> the training data is drawn from Western Spotify '
    f'playlists across six genres. Only 5 of the 28,356 tracks are by Indian artists, so '
    f'the model has effectively never heard Bollywood, Punjabi or any Indian classical '
    f'music. Its predictions should not be trusted for them.</p>',
    unsafe_allow_html=True)
