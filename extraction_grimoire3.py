#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PROTOCOLE D'EXTRACTION : GRIMOIRE DU GLITCH               ║
║                    Version 3.0 — Text Mining & DataViz étendus               ║
║                                                                              ║
║  "La réalité est une solution triviale d'un système non-linéaire.            ║
║   Les Dieux sont les solutions idéales dans la mousse quantique."            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Nouveautés v3 (voir aussi le récapitulatif fourni avec ce script) :
  • Corpus structuré (dépôt / fichier / passage) au lieu d'un énorme texte concaténé
  • Clone parallèle, historique git (commits), dépôts homonymes distincts (owner__nom)
  • Un seul passage spaCy : lemmes, NER, graphe, collocations, stats lexicales
  • Mots-clés sémantiques corrigés (embeddings multilingues + MMR, sans troncature)
  • Termes distinctifs par dépôt (log-odds à prior informatif), LDA avec k auto (cohérence NPMI)
  • Thèmes sémantiques (clustering d'embeddings) + carte 2D
  • Graphe : backbone, k-core, participation, communautés nommées, références inter-dépôts
  • DataViz : ~20 figures PNG + dashboard HTML interactif (Plotly) + graphe pyvis
  • Exports : JSON (compatible v2), CSV, GraphML, GRIMOIRE_RESUME.md (prêt pour un LLM)

Usage :
    python extraction_grimoire3.py                 # run complet
    python extraction_grimoire3.py --force         # re-clone tout
    python extraction_grimoire3.py --no-history    # clones shallow (rapide, sans timeline)
    python extraction_grimoire3.py --no-embeddings # sans sentence-transformers
    python extraction_grimoire3.py --help
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import html as html_lib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import tokenize
import traceback
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

# On ne coupe QUE les warnings bruyants connus (l'ancienne version masquait tout)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*token_pattern.*")
warnings.filterwarnings("ignore", message=".*stop_words.*")

# ─── Stack scientifique (obligatoire) ──────────────────────────────────────────
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # pas de GUI : on écrit des fichiers
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import (
    CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer)
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# ─── Enrichissements optionnels (dégradation gracieuse) ────────────────────────
def _try_import(name):
    try:
        return __import__(name, fromlist=["_"])
    except Exception:
        return None

spacy = _try_import("spacy")
HAS_SPACY = spacy is not None
try:
    from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOP
except Exception:  # spaCy absent : mini liste de secours
    FR_STOP = set("""le la les un une des du de d l et ou mais donc or ni car que qui quoi dont où
    ce cet cette ces mon ma mes ton ta tes son sa ses notre nos votre vos leur leurs je tu il elle on
    nous vous ils elles en y à au aux par pour sur sous dans avec sans entre vers chez est sont été
    être a ont avoir fait faire plus moins très pas ne se si comme aussi alors ainsi""".split())

sentence_transformers = _try_import("sentence_transformers")
HAS_ST = sentence_transformers is not None
wordcloud_mod = _try_import("wordcloud")
HAS_WORDCLOUD = wordcloud_mod is not None
pyvis_network = _try_import("pyvis.network")
HAS_PYVIS = pyvis_network is not None
plotly_go = _try_import("plotly.graph_objects")
HAS_PLOTLY = plotly_go is not None
umap_mod = _try_import("umap")
HAS_UMAP = umap_mod is not None and hasattr(umap_mod, "UMAP")
adjust_text_mod = _try_import("adjustText")
HAS_ADJUST = adjust_text_mod is not None and hasattr(adjust_text_mod, "adjust_text")
louvain_mod = _try_import("community")  # python-louvain (facultatif : networkx suffit)
HAS_LOUVAIN = louvain_mod is not None and hasattr(louvain_mod, "best_partition")


# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE FENG-SHUI — Terminal Colorisé
# ══════════════════════════════════════════════════════════════════════════════

class C:
    """Palette ANSI harmonisée (désactivée automatiquement hors terminal / NO_COLOR)."""
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"; ITALIC = "\033[3m"
    UNDER = "\033[4m"; BLINK = "\033[5m"; REVERSE = "\033[7m"
    BLACK = "\033[30m"; RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    BLUE = "\033[34m"; MAGENTA = "\033[35m"; CYAN = "\033[36m"; WHITE = "\033[37m"
    BRED = "\033[91m"; BGREEN = "\033[92m"; BYELLOW = "\033[93m"; BBLUE = "\033[94m"
    BMAGENTA = "\033[95m"; BCYAN = "\033[96m"; BWHITE = "\033[97m"

    @classmethod
    def disable(cls):
        for k, v in list(vars(cls).items()):
            if k.isupper() and isinstance(v, str):
                setattr(cls, k, "")


if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
    C.disable()
elif os.name == "nt":
    os.system("")  # active les séquences ANSI sous Windows 10+

# Style global matplotlib "cosmique"
plt.rcParams.update({
    "figure.facecolor": "#0a0a12", "axes.facecolor": "#0a0a12",
    "savefig.facecolor": "#0a0a12", "text.color": "#d0d0e0",
    "axes.labelcolor": "#d0d0e0", "xtick.color": "#8080a0", "ytick.color": "#8080a0",
    "axes.edgecolor": "#2a2a3a", "grid.color": "#1a1a2a", "font.family": "monospace",
    "axes.titlecolor": "#d0d0e0",
})

COSMIC_CMAP = LinearSegmentedColormap.from_list(
    "cosmic", ["#0a0a12", "#1a1040", "#4a2080", "#a040c0", "#ff60a0", "#ffd060"])
NEON_CMAP = LinearSegmentedColormap.from_list(
    "neon", ["#00fff7", "#00b4ff", "#7a5cff", "#ff5cc8", "#ffb45c"])
# Palette catégorielle lisible sur fond sombre (cyclable, contrairement à "hsv")
PALETTE = ["#00fff7", "#ff60a0", "#ffb45c", "#7a5cff", "#5cff7a", "#00b4ff", "#ff5c5c",
           "#c0c0ff", "#e0ff5c", "#ff8cf0", "#5cffd0", "#ffa07a", "#9ad0ff", "#d0a0ff"]


def couleur_idx(i: int) -> str:
    return PALETTE[int(i) % len(PALETTE)] if i is not None and i >= 0 else "#606080"


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLE FENG-SHUI — Helpers d'affichage
# ══════════════════════════════════════════════════════════════════════════════

def _term_width():
    return shutil.get_terminal_size((100, 30)).columns

def banner(titre, sous_titre="", couleur=C.BCYAN, symbole="✦"):
    w = min(_term_width(), 100)
    print(f"\n{couleur}{'═' * w}{C.RESET}")
    if sous_titre:
        pad = (w - len(titre) - len(sous_titre) - 5) // 2
        print(f"{couleur}{symbole}{C.RESET}  {C.BOLD}{C.BWHITE}{titre}{C.RESET}"
              f"{' ' * max(pad, 1)}{C.DIM}{C.ITALIC}{sous_titre}{C.RESET}  {couleur}{symbole}{C.RESET}")
    else:
        pad = (w - len(titre) - 4) // 2
        print(f"{' ' * max(pad, 0)}{couleur}{symbole}{C.RESET}  "
              f"{C.BOLD}{C.BWHITE}{titre}{C.RESET}  {couleur}{symbole}{C.RESET}")
    print(f"{couleur}{'═' * w}{C.RESET}")

def section(titre, couleur=C.BMAGENTA, icone="◈"):
    print(f"\n{couleur}{icone} {C.BOLD}{titre}{C.RESET}")
    print(f"{couleur}{'─' * min(len(titre) + 4, _term_width())}{C.RESET}")

_LOG_STYLES = {
    "info": (C.BCYAN, "›"), "ok": (C.BGREEN, "✓"), "warn": (C.BYELLOW, "⚠"),
    "err": (C.BRED, "✗"), "star": (C.BMAGENTA, "✧"), "run": (C.BBLUE, "⟳"),
    "spark": (C.BYELLOW, "✦"), "sleep": (C.DIM, "·"),
}

def log(msg, niveau="info", indent=2):
    col, ic = _LOG_STYLES.get(niveau, (C.WHITE, "·"))
    print(f"{' ' * indent}{col}{ic}{C.RESET} {msg}")

def progress_bar(current, total, prefix="", width=40, couleur=C.BCYAN):
    ratio = current / total if total else 0
    filled = int(width * ratio)
    bar = f"{couleur}{'█' * filled}{C.DIM}{'░' * (width - filled)}{C.RESET}"
    print(f"\r  {prefix} {bar} {C.BWHITE}{ratio * 100:5.1f}%{C.RESET}", end="", flush=True)
    if current >= total:
        print()

def kv_line(cle, valeur, couleur_cle=C.CYAN, couleur_val=C.BWHITE, indent=4):
    print(f"{' ' * indent}{couleur_cle}{cle:<32}{C.RESET} {C.DIM}│{C.RESET} "
          f"{couleur_val}{valeur}{C.RESET}")

def hr(couleur=C.DIM, char="┄"):
    print(f"{couleur}{char * min(_term_width(), 100)}{C.RESET}")

def sparkline(valeurs, largeur=40):
    if not len(valeurs):
        return ""
    blocs = "▁▂▃▄▅▆▇█"
    mn, mx = min(valeurs), max(valeurs)
    if mx == mn:
        return blocs[0] * min(len(valeurs), largeur)
    idx = [int((v - mn) / (mx - mn) * (len(blocs) - 1)) for v in valeurs]
    return f"{C.BCYAN}{''.join(blocs[i] for i in idx[-largeur:])}{C.RESET}"

def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", " ")

DEBUG = False
PHASE_TIMES: dict[str, float] = {}

def safe(nom, fn, *args, default=None, **kw):
    """Exécute une étape ; en cas d'erreur : log + on continue (sauf --debug)."""
    t0 = time.time()
    try:
        return fn(*args, **kw)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        log(f"{nom} : {type(e).__name__}: {str(e)[:160]}", "err")
        if DEBUG:
            traceback.print_exc()
        return default
    finally:
        PHASE_TIMES[nom] = PHASE_TIMES.get(nom, 0.0) + time.time() - t0


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

REPOS = [
    "https://github.com/doktornand/Fracturo2075",
    "https://github.com/doktornand/Lingua-nostra-mundum-format",
    "https://github.com/doktornand/Corpus",
    "https://github.com/doktornand/2075-Code",
    "https://github.com/doktornand/FractalTime-TimeWaveZero",
    "https://github.com/doktornand/BDO",
    "https://github.com/doktornand/FGlitch",
    "https://github.com/JinroOe/Europropagate",
    "https://github.com/Syrinx-2112/Discordian-Heliger",
    "https://github.com/Syrinx-2112/Chronos-Trame",
    "https://github.com/Syrinx-2112/Trame-Watcher",
    "https://github.com/KareyPyer/Uncover-Echo",
    "https://github.com/KareyPyer/Shodan-Echo",
    "https://github.com/KareyPyer/Latent-Fracturo-Studio",
    "https://github.com/KareyPyer/Phobos-Omega-Noir",
    "https://github.com/KareyPyer/Meme-Forge",
    "https://github.com/KareyPyer/Meme-Prompt-Studio",
    "https://github.com/KareyPyer/Vauvillensis-2075",
    "https://github.com/V-Demon/Fulda93",
    "https://github.com/V-Demon/Chronos-Trame",
    "https://github.com/V-Demon/Cache-Echo",
    "https://github.com/V-Demon/Uncover-Echo",
    "https://github.com/V-Demon/Shodan-Echo",
    "https://github.com/V-Demon/MTT2075---Carnage-de-Broceliande",
    "https://github.com/V-Demon/Tech2075",
    "https://github.com/V-Demon/Geopolitique2075",
]

LEXIQUE_TRAME = {
    "delta", "trame", "strate", "runes", "fracturoscript", "echo-guillaume",
    "caen-profonde", "convergence", "mousse quantique", "holographique",
    "non-linéaire", "solution idéale", "solution triviale", "paléo-mème",
    "hackervölva", "sans-marque", "codex", "vauvillensis", "nidification",
    "glitch", "fracturo", "chronos", "shodan", "echo", "mème",
}

EXTENSIONS_TEXTE = {".md", ".txt", ".rst", ".org", ".py", ".json", ".yaml", ".yml", ".ipynb"}
EXCLUSIONS_DIR = {".git", ".github", "node_modules", "__pycache__", ".venv", "venv",
                  "dist", "build", ".idea", ".vscode", "site-packages"}
FICHIERS_IGNORES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
                    "composer.lock", "license", "license.md", "license.txt"}
NER_LABELS = {"PER", "PERSON", "LOC", "GPE", "ORG", "MISC", "EVENT", "WORK_OF_ART"}
NOISE_WORDS = {"http", "https", "www", "com", "org", "github", "readme", "md", "txt", "py",
               "json", "yaml", "yml", "true", "false", "none", "null", "def", "return", "self",
               "import", "from", "class", "print", "todo", "fixme", "etc", "via", "ex", "cf"}
STOP = set(FR_STOP) | set(ENGLISH_STOP_WORDS) | NOISE_WORDS


def _canon(terme: str) -> str:
    return re.sub(r"[\s\-]+", "_", terme.strip().lower())

# "mousse quantique" / "echo-guillaume" etc. ne pouvaient JAMAIS matcher un token spaCy dans la v2.
# On les fusionne en un token unique (mousse_quantique) avant l'analyse.
LEXIQUE_CANON = {_canon(t): t for t in LEXIQUE_TRAME}
def _motif_terme(terme: str):
    morceaux = [re.escape(m) for m in re.split(r"[\s\-]+", terme.strip()) if m]
    return re.compile(r"(?<!\w)" + r"[\s\-]+".join(morceaux) + r"(?!\w)", re.I)

LEXIQUE_PATTERNS = sorted(
    [(_motif_terme(t), _canon(t)) for t in LEXIQUE_TRAME if re.search(r"[\s\-]", t)],
    key=lambda x: -len(x[1]))
LEXIQUE_SET = set(LEXIQUE_CANON)


def pretty(terme: str) -> str:
    """mousse_quantique -> mousse quantique (affichage)."""
    return terme.replace("_", " ")


@dataclass
class Config:
    repos: list = field(default_factory=lambda: list(REPOS))
    work_dir: Path = Path("./sanctuaire_grimoire")
    cache_dir: Path = Path("./.grimoire_cache")
    output_dir: Path = Path("./grimoire_output")
    cache_ttl_h: float = 24.0
    force: bool = False
    history: bool = True            # clone "blobless" avec historique (commits)
    workers: int = 6                # clones parallèles
    max_file_bytes: int = 1_500_000
    passage_words: int = 120        # taille cible d'un passage
    max_passages: int = 40_000
    max_embed: int = 6_000
    n_procs: int = 1                # processus spaCy
    min_cooc: int = 2
    graph_viz_nodes: int = 220
    embeddings: bool = True
    embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    n_topics: int = 0               # 0 = choix automatique
    offline_plotly: bool = False    # True = plotly.js embarqué (HTML ~4 Mo, 100 % offline)
    seed: int = 42


CFG = Config()


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — INITIALISATION DU SANCTUAIRE (clone parallèle + historique git)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RepoInfo:
    url: str
    owner: str
    name: str
    path: Path
    ok: bool = False
    cached: bool = False
    head: str = ""
    error: str = ""
    shallow: bool = True
    label: str = ""

    @property
    def key(self) -> str:
        return f"{self.owner}__{self.name}"

    @property
    def full(self) -> str:
        return f"{self.owner}/{self.name}"


def parse_repo_url(url: str):
    """URL -> (owner, nom). Les dépôts homonymes (Chronos-Trame chez 2 owners) restent distincts.
    (v2 : clone dans WORK_DIR/<nom> => le 2e écrasait le 1er, 3 dépôts perdus sur 26.)"""
    u = url.strip().rstrip("/")
    if u.endswith(".git"):
        u = u[:-4]
    parts = [p for p in re.split(r"[/:\\]", u) if p]
    return (parts[-2] if len(parts) >= 2 else "local"), parts[-1]


def attribuer_labels(repos: list[RepoInfo]):
    """Label court : 'nom' si unique, sinon 'nom (owner)'."""
    cnt = Counter(r.name for r in repos)
    for r in repos:
        r.label = r.name if cnt[r.name] == 1 else f"{r.name} ({r.owner})"


def _git(args, cwd=None, timeout=240):
    # GIT_TERMINAL_PROMPT=0 : un dépôt supprimé/privé ne doit pas bloquer sur "Username for github.com"
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    return subprocess.run(["git", *args], cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                          timeout=timeout, env=env)


def _stamp_path(r: RepoInfo) -> Path:
    return r.path / ".grimoire_stamp"


def _lire_stamp(r: RepoInfo):
    try:
        return json.loads(_stamp_path(r).read_text())
    except Exception:
        return None


def _cache_frais(r: RepoInfo) -> bool:
    st = _lire_stamp(r)
    if not st or not r.path.exists():
        return False
    if CFG.history and st.get("shallow", True):
        return False  # on veut l'historique : un clone shallow ne convient plus
    return (time.time() - st.get("time", 0)) / 3600 < CFG.cache_ttl_h


def _cloner(r: RepoInfo):
    if not CFG.force and _cache_frais(r):
        r.ok, r.cached = True, True
        st = _lire_stamp(r) or {}
        r.head, r.shallow = st.get("head", ""), st.get("shallow", True)
        return r
    if r.path.exists():
        shutil.rmtree(r.path, ignore_errors=True)
    tentatives = ([["clone", "--filter=blob:none", "--quiet", r.url, str(r.path)]]
                  if CFG.history else [])
    tentatives.append(["clone", "--depth", "1", "--quiet", r.url, str(r.path)])
    for i, args in enumerate(tentatives):
        try:
            p = _git(args)
            if p.returncode == 0 and r.path.exists():
                r.ok = True
                r.shallow = "--depth" in args
                break
            lignes = [l for l in p.stderr.decode(errors="ignore").splitlines() if l.strip()]
            r.error = next((l for l in lignes if "fatal" in l.lower()), lignes[0] if lignes else
                           f"git code {p.returncode}")[:90]
            shutil.rmtree(r.path, ignore_errors=True)
        except subprocess.TimeoutExpired:
            r.error = "timeout"
            shutil.rmtree(r.path, ignore_errors=True)
        except Exception as e:
            r.error = str(e)[:90]
    if r.ok:
        try:
            r.head = _git(["rev-parse", "HEAD"], cwd=r.path).stdout.decode().strip()
        except Exception:
            pass
        try:
            _stamp_path(r).write_text(json.dumps(
                {"time": time.time(), "head": r.head, "shallow": r.shallow}))
        except Exception:
            pass
    return r


def lire_commits(r: RepoInfo) -> list[dict]:
    """Métadonnées des commits (sans diff => marche sur un clone blobless)."""
    try:
        p = _git(["log", "-n", "20000", "--pretty=format:%H%x1f%an%x1f%aI%x1f%s"], cwd=r.path)
    except Exception:
        return []
    out = []
    for ligne in p.stdout.decode("utf-8", errors="replace").splitlines():
        champs = ligne.split("\x1f")
        if len(champs) < 4:
            continue
        sha, auteur, date, sujet = champs[0], champs[1], champs[2], "\x1f".join(champs[3:])
        try:
            dt = pd.Timestamp(date).tz_convert("UTC")
        except Exception:
            continue
        out.append({"repo": r.label, "sha": sha[:10], "auteur": auteur, "date": dt, "sujet": sujet})
    return out


def initialiser_sanctuaire() -> tuple[list[RepoInfo], pd.DataFrame]:
    banner("PHASE I — INITIALISATION DU SANCTUAIRE", "Ingestion des strates GitHub",
           couleur=C.BCYAN, symbole="❖")
    CFG.work_dir.mkdir(parents=True, exist_ok=True)
    CFG.cache_dir.mkdir(parents=True, exist_ok=True)
    CFG.output_dir.mkdir(parents=True, exist_ok=True)

    vus, repos = set(), []
    for url in CFG.repos:
        owner, name = parse_repo_url(url)
        if (owner, name) in vus:
            continue
        vus.add((owner, name))
        r = RepoInfo(url=url, owner=owner, name=name, path=CFG.work_dir / f"{owner}__{name}")
        repos.append(r)
    attribuer_labels(repos)

    total = len(repos)
    log(f"{total} dépôts • {CFG.workers} clones en parallèle • "
        f"mode {'historique (blobless)' if CFG.history else 'shallow (depth 1)'}", "run")
    done = 0
    progress_bar(0, total, prefix="Clonage")
    with ThreadPoolExecutor(max_workers=max(1, CFG.workers)) as ex:
        futs = [ex.submit(_cloner, r) for r in repos]
        for _ in as_completed(futs):
            done += 1
            progress_bar(done, total, prefix="Clonage")

    ok = [r for r in repos if r.ok]
    ko = [r for r in repos if not r.ok]
    print()
    kv_line("Dépôts clonés", str(sum(1 for r in ok if not r.cached)), couleur_val=C.BGREEN)
    kv_line("Dépôts en cache", str(sum(1 for r in ok if r.cached)), couleur_val=C.BCYAN)
    kv_line("Échecs", str(len(ko)), couleur_val=C.BRED if ko else C.DIM)
    if ko:
        section("Dépôts inaccessibles", couleur=C.BYELLOW, icone="⚠")
        for r in ko[:12]:
            log(f"{r.full:<45} {C.DIM}{r.error}{C.RESET}", "warn", indent=4)

    commits = []
    for r in ok:
        commits += lire_commits(r)
    cdf = pd.DataFrame(commits, columns=["repo", "sha", "auteur", "date", "sujet"])
    if len(cdf):
        kv_line("Commits lus", fmt_int(len(cdf)))
        kv_line("Période", f"{cdf['date'].min():%Y-%m-%d} → {cdf['date'].max():%Y-%m-%d}")
    log(f"Sanctuaire stabilisé — {len(ok)} strates actives.", "ok")
    return ok, cdf


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — MOISSON DE LA SUBSTANCE (extracteurs par format, provenance conservée)
# ══════════════════════════════════════════════════════════════════════════════

RE_FENCE = re.compile(r"```.*?```|~~~.*?~~~", re.S)
RE_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
RE_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
RE_URL = re.compile(r"(?:https?://|www\.)\S+")
RE_HTML = re.compile(r"<[^>]{1,200}>")
RE_HEAD = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.M)
RE_TABLE = re.compile(r"^\s*\|?[\s:\-|]{3,}\|?\s*$", re.M)
RE_TODO = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b[:\s]*(.{0,120})")
RE_TASK = re.compile(r"^\s*[-*]\s+\[( |x|X)\]\s+(.{3,120})$", re.M)
RE_CODEISH = re.compile(r"^(import |from \S+ import|def |class |return\b|if |elif |else:|for |while |"
                        r"print\(|self\.|[\w\.\[\]]+\s*[\+\-\*/]?=\s*\S|\w+\([^)]*\)\s*$)")
RE_WORD = re.compile(r"[A-Za-zÀ-ÿŒœ][A-Za-zÀ-ÿŒœ'’\-_]*")


def nettoyer_markdown(txt: str):
    """Texte propre (sans code/URL/badges) + liste des titres."""
    titres = [t.strip() for t in RE_HEAD.findall(txt)]
    txt = RE_FENCE.sub(" ", txt)
    txt = RE_IMG.sub(" ", txt)
    txt = RE_LINK.sub(r"\1", txt)
    txt = RE_URL.sub(" ", txt)
    txt = RE_HTML.sub(" ", txt)
    txt = RE_TABLE.sub(" ", txt)
    txt = txt.replace("`", "").replace("|", " ")
    txt = re.sub(r"^\s{0,3}#{1,6}\s*", "", txt, flags=re.M)
    txt = re.sub(r"[*_]{1,3}(\S[^*_\n]*?)[*_]{1,3}", r"\1", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip(), titres


def _phrase_utile(s: str, min_mots=6, ratio=0.6) -> bool:
    mots = RE_WORD.findall(s)
    if len(mots) < min_mots:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in s)
    return alpha / max(len(s), 1) >= ratio


def extraire_python(content: str):
    """Docstrings (AST) + commentaires (tokenize, plus fiable que la regex '#') +
    chaînes littérales 'prose' (prompts, lore embarqué) + métriques de code."""
    metr = {"loc": sum(1 for l in content.splitlines() if l.strip()),
            "fonctions": 0, "classes": 0, "docstrings": 0, "complexite": 0, "imports": []}
    morceaux, todos, doc_texts = [], [], set()
    try:
        tree = ast.parse(content)
        defs = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if not isinstance(node, ast.Module):
                    defs += 1
                    metr["classes" if isinstance(node, ast.ClassDef) else "fonctions"] += 1
                d = ast.get_docstring(node)
                if d:
                    doc_texts.add(d.strip())
                    morceaux.append(d)
                    if not isinstance(node, ast.Module):
                        metr["docstrings"] += 1
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler,
                                   ast.BoolOp, ast.comprehension, ast.IfExp)):
                metr["complexite"] += 1
            elif isinstance(node, ast.Import):
                metr["imports"] += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                metr["imports"].append(node.module.split(".")[0])
        for node in ast.walk(tree):  # chaînes "prose"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value.strip()
                if s not in doc_texts and len(s) > 50 and _phrase_utile(s, 8):
                    morceaux.append(s)
        metr["doc_cov"] = round(metr["docstrings"] / defs, 3) if defs else None
    except (SyntaxError, ValueError, RecursionError):
        metr["doc_cov"] = None
    try:
        for tok in tokenize.generate_tokens(io.StringIO(content).readline):
            if tok.type != tokenize.COMMENT:
                continue
            c = tok.string.lstrip("#").strip()
            if c.lower().startswith(("noqa", "type:", "pylint", "fmt:", "-*-", "!/", "coding")):
                continue
            m = RE_TODO.search(c)
            if m:
                todos.append((m.group(1), m.group(2).strip()))
            if RE_CODEISH.match(c) or not _phrase_utile(c, 3, 0.6):
                continue
            morceaux.append(c)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return "\n\n".join(morceaux), metr, todos


def _chaines_json(obj, out, profondeur=0):
    if profondeur > 12 or len(out) > 20000:
        return
    if isinstance(obj, str):
        s = obj.strip()
        if len(s) >= 15 and " " in s and _phrase_utile(s, 3, 0.6):
            out.append(s)
    elif isinstance(obj, dict):
        for v in obj.values():
            _chaines_json(v, out, profondeur + 1)
    elif isinstance(obj, list):
        for v in obj:
            _chaines_json(v, out, profondeur + 1)


def extraire_json(content: str, ext: str) -> str:
    try:
        data = json.loads(content)
    except Exception:
        return ""
    if ext == ".ipynb" and isinstance(data, dict):
        cells = [("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                  else str(c.get("source", ""))) for c in data.get("cells", [])
                 if c.get("cell_type") == "markdown"]
        return "\n\n".join(nettoyer_markdown(c)[0] for c in cells)
    out: list[str] = []
    _chaines_json(data, out)
    return "\n\n".join(out)


def extraire_yaml(content: str) -> str:
    out = []
    for l in content.splitlines():
        s = l.strip()
        if s.startswith("#"):
            s = s.lstrip("# ").strip()
        else:
            m = re.match(r"^[-\s]*[\w\-\"']+:\s*(.+)$", s) or re.match(r"^-\s+(.+)$", s)
            s = m.group(1).strip(" \"'") if m else ""
        if s and _phrase_utile(s, 3, 0.6):
            out.append(s)
    return "\n".join(out)


def detecter_langue(texte: str) -> str:
    mots = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", texte[:20000].lower())[:3000]
    if len(mots) < 15:
        return "?"
    fr = sum(m in FR_STOP and m not in ENGLISH_STOP_WORDS for m in mots)
    en = sum(m in ENGLISH_STOP_WORDS and m not in FR_STOP for m in mots)
    if fr > en * 1.3:
        return "fr"
    if en > fr * 1.3:
        return "en"
    return "mixte"


def _lire_fichier(fp: Path):
    try:
        if fp.stat().st_size > CFG.max_file_bytes:
            return None
        raw = fp.read_bytes()
    except Exception:
        return None
    if b"\x00" in raw[:2048]:
        return None
    return raw.decode("utf-8", errors="replace")


@dataclass
class Corpus:
    docs: list = field(default_factory=list)          # documents dédupliqués (provenance incluse)
    stats: dict = field(default_factory=dict)         # par extension
    inventaire: pd.DataFrame = None                   # tous fichiers (taille par repo/ext)
    doublons: pd.DataFrame = None                     # fichiers identiques entre dépôts
    code: pd.DataFrame = None                         # métriques Python
    todos: pd.DataFrame = None
    references: pd.DataFrame = None                   # mentions inter-dépôts
    hash_par_repo: dict = field(default_factory=dict)
    titres: list = field(default_factory=list)
    erreurs: int = 0


def _motif_references(repos: list[RepoInfo]):
    noms = sorted({r.name for r in repos if len(r.name) >= 4}, key=len, reverse=True)
    if not noms:
        return None
    alt = "|".join(re.escape(n) for n in noms)
    return re.compile(r"(?<![\w\-])(" + alt + r")(?![\w\-]*[A-Za-z0-9])", re.I)


def moissonner_la_substance(repos: list[RepoInfo]) -> Corpus:
    banner("PHASE II — MOISSON DE LA SUBSTANCE", "Extraction multi-format",
           couleur=C.BMAGENTA, symbole="❖")
    cor = Corpus()
    stats = defaultdict(lambda: {"fichiers": 0, "octets": 0, "extrait": 0})
    inv, code_rows, todo_rows, ref_rows, dup_rows = [], [], [], [], []
    vus: dict[str, tuple] = {}
    motif_ref = _motif_references(repos)
    noms_par_lower = defaultdict(list)
    for r in repos:
        noms_par_lower[r.name.lower()].append(r)

    for idx, r in enumerate(repos, 1):
        progress_bar(idx - 1, len(repos), prefix=f"[{idx:>2}/{len(repos)}] {r.label[:28]:<28}")
        hashes = set()
        for root, dirs, files in os.walk(r.path):
            dirs[:] = [d for d in dirs if d not in EXCLUSIONS_DIR and not d.startswith(".")]
            for f in files:
                fp = Path(root) / f
                ext = fp.suffix.lower() or "(aucune)"
                try:
                    taille = fp.stat().st_size
                except OSError:
                    continue
                if f.startswith(".grimoire_"):
                    continue
                inv.append((r.label, ext, taille))
                if ext not in EXTENSIONS_TEXTE or f.lower() in FICHIERS_IGNORES or ".min." in f.lower():
                    continue
                content = _lire_fichier(fp)
                if content is None or not content.strip():
                    continue
                rel = str(fp.relative_to(r.path))
                try:
                    titres, metr = [], None
                    if ext == ".py":
                        txt, metr, tds = extraire_python(content)
                        for kind, t in tds:
                            todo_rows.append((r.label, rel, kind, t))
                    elif ext in (".md", ".rst", ".org", ".txt"):
                        if ext == ".md":
                            txt, titres = nettoyer_markdown(content)
                        else:
                            txt = RE_URL.sub(" ", content)
                        for m in RE_TODO.finditer(content):
                            todo_rows.append((r.label, rel, m.group(1), m.group(2).strip()))
                        for m in RE_TASK.finditer(content):
                            if m.group(1) == " ":
                                todo_rows.append((r.label, rel, "TÂCHE", m.group(2).strip()))
                    elif ext in (".json", ".ipynb"):
                        txt = extraire_json(content, ext)
                    else:
                        txt = extraire_yaml(content)
                    # références inter-dépôts (sur le contenu brut : URLs et code compris)
                    if motif_ref:
                        for m in motif_ref.finditer(content):
                            for cible in noms_par_lower.get(m.group(1).lower(), []):
                                if cible.key != r.key:
                                    ref_rows.append((r.label, cible.label))
                    if not txt or len(txt.strip()) < 20:
                        continue
                    h = hashlib.sha1(re.sub(r"\s+", " ", txt).strip().lower().encode()).hexdigest()
                    hashes.add(h)
                    if h in vus:
                        dup_rows.append((vus[h][0], vus[h][1], r.label, rel, len(txt)))
                        continue
                    vus[h] = (r.label, rel)
                    cor.docs.append({
                        "id": len(cor.docs), "repo": r.label, "owner": r.owner, "path": rel,
                        "ext": ext, "texte": txt, "octets": taille, "langue": detecter_langue(txt),
                        "nom": f.lower(),
                    })
                    cor.titres += [(r.label, rel, t) for t in titres]
                    if metr:
                        code_rows.append({"repo": r.label, "path": rel, **{k: v for k, v in metr.items()
                                                                          if k != "imports"},
                                          "imports": ",".join(sorted(set(metr["imports"])))})
                    stats[ext]["fichiers"] += 1
                    stats[ext]["octets"] += taille
                    stats[ext]["extrait"] += len(txt)
                except Exception:
                    cor.erreurs += 1
        cor.hash_par_repo[r.label] = hashes
    progress_bar(len(repos), len(repos), prefix="Moisson terminée")

    cor.stats = {k: dict(v) for k, v in stats.items()}
    cor.inventaire = pd.DataFrame(inv, columns=["repo", "ext", "octets"])
    cor.doublons = pd.DataFrame(dup_rows, columns=["repo_orig", "path_orig", "repo_dup", "path_dup", "car"])
    cor.code = pd.DataFrame(code_rows)
    cor.todos = pd.DataFrame(todo_rows, columns=["repo", "path", "type", "texte"])
    refs = pd.DataFrame(ref_rows, columns=["source", "cible"])
    cor.references = (refs.groupby(["source", "cible"]).size().reset_index(name="mentions")
                      if len(refs) else pd.DataFrame(columns=["source", "cible", "mentions"]))

    total_car = sum(len(d["texte"]) for d in cor.docs)
    print()
    kv_line("Caractères extraits", fmt_int(total_car))
    kv_line("Mots (approx.)", fmt_int(sum(len(d["texte"].split()) for d in cor.docs)))
    kv_line("Documents uniques", fmt_int(len(cor.docs)))
    kv_line("Doublons exacts écartés", fmt_int(len(cor.doublons)))
    kv_line("Erreurs I/O", str(cor.erreurs), couleur_val=C.BYELLOW if cor.erreurs else C.DIM)
    langs = Counter(d["langue"] for d in cor.docs)
    kv_line("Langues (docs)", " ".join(f"{k}:{v}" for k, v in langs.most_common()))

    section("Répartition par extension", C.BCYAN, "▤")
    for ext, s in sorted(cor.stats.items(), key=lambda x: -x[1]["extrait"]):
        pct = s["extrait"] / max(total_car, 1) * 100
        bar = f"{C.BCYAN}{'█' * int(pct / 2)}{C.DIM}{'░' * (50 - int(pct / 2))}{C.RESET}"
        print(f"    {ext:<8} {s['fichiers']:>4} fichiers │ {s['extrait']:>9,} car. │ {bar} {pct:5.1f}%")

    section("Top 10 dépôts les plus denses (caractères utiles)", C.BMAGENTA, "▤")
    dens = Counter()
    for d in cor.docs:
        dens[d["repo"]] += len(d["texte"])
    for nom, n in dens.most_common(10):
        print(f"    {nom:<44} {n:>9,} car.  {C.BMAGENTA}{'▓' * min(int(n / max(dens.values()) * 30), 30)}{C.RESET}")
    if len(cor.doublons):
        section("Doublons entre dépôts (forks / copies)", C.BYELLOW, "⚠")
        top = cor.doublons.groupby(["repo_orig", "repo_dup"]).size().sort_values(ascending=False).head(6)
        for (a, b), n in top.items():
            log(f"{a}  ⇄  {b} : {n} fichiers identiques", "warn", indent=4)
    log(f"{fmt_int(total_car)} caractères de substance extraits.", "ok")
    return cor


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3a — PASSAGES + PASSE LINGUISTIQUE UNIQUE (lemmes, NER, graphe, collocations)
# ══════════════════════════════════════════════════════════════════════════════

def canon_text(texte: str) -> str:
    """Fusionne les termes du lexique multi-mots / à tiret en un token unique (mousse_quantique)."""
    for pat, rep in LEXIQUE_PATTERNS:
        texte = pat.sub(rep, texte)
    return texte


def decouper_passages(docs: list[dict]) -> list[dict]:
    """Découpe chaque document en passages (~passage_words mots) en respectant les paragraphes.
    Unité d'analyse de tout le pipeline (LDA, embeddings, graphe) : bien meilleure que 'phrases'
    (trop courtes) ou 'texte entier' (KeyBERT tronquait à ~256 tokens)."""
    cible = CFG.passage_words
    passages = []
    for d in docs:
        unites = []
        for para in re.split(r"\n\s*\n", d["texte"]):
            para = para.strip()
            if not para:
                continue
            if len(para.split()) > 2 * cible:
                unites += [s.strip() for s in re.split(r"(?<=[.!?…])\s+|\n", para) if s.strip()]
            else:
                unites.append(para)
        buf, n = [], 0
        def flush():
            nonlocal buf, n
            if buf and n >= 6:
                passages.append({"pid": len(passages), "doc_id": d["id"], "repo": d["repo"],
                                 "ext": d["ext"], "langue": d["langue"], "path": d["path"],
                                 "texte": "\n".join(buf)})
            buf, n = [], 0
        for u in unites:
            w = len(u.split())
            if buf and n + w > cible:
                flush()
            buf.append(u)
            n += w
        flush()

    if len(passages) > CFG.max_passages:  # échantillonnage stratifié par dépôt (les petits dépôts survivent)
        rng = random.Random(CFG.seed)
        par_repo = defaultdict(list)
        for p in passages:
            par_repo[p["repo"]].append(p)
        ratio = CFG.max_passages / len(passages)
        gardes = []
        for lst in par_repo.values():
            k = min(len(lst), max(int(len(lst) * ratio), 300))
            gardes += rng.sample(lst, k)
        gardes.sort(key=lambda p: p["pid"])
        log(f"Échantillonnage stratifié : {len(passages)} → {len(gardes)} passages", "warn")
        passages = gardes
        for i, p in enumerate(passages):
            p["pid"] = i
    return passages


def _charger_spacy():
    """Charge un vrai modèle français (lemmes + POS + NER). None => mode de secours regex."""
    if not HAS_SPACY:
        log("spaCy absent : mode de secours (pip install spacy)", "warn")
        return None
    for modele in ("fr_core_news_lg", "fr_core_news_md", "fr_core_news_sm"):
        try:
            nlp = spacy.load(modele)
        except Exception:
            continue
        # senter à la place du parser : ~2x plus rapide, suffisant pour segmenter en phrases
        try:
            if nlp.has_pipe("parser") and "senter" in nlp.component_names:
                nlp.enable_pipe("senter")
                nlp.disable_pipe("parser")
        except Exception:
            pass
        if not (nlp.has_pipe("parser") or nlp.has_pipe("senter")):
            nlp.add_pipe("sentencizer")
        for tok in LEXIQUE_CANON:  # le tokenizer ne doit pas casser mousse_quantique
            try:
                nlp.tokenizer.add_special_case(tok, [{"ORTH": tok}])
            except Exception:
                pass
        log(f"Modèle spaCy chargé : {C.BWHITE}{modele}", "ok")
        return nlp
    log("Aucun modèle spaCy FR. → python -m spacy download fr_core_news_md  (mode de secours regex)", "warn")
    return None


def _nettoyer_entite(txt: str):
    txt = re.sub(r"\s+", " ", txt).strip(" \t\n\"'«»“”()[]{}.,;:!?*_-")
    if not (2 <= len(txt) <= 60) or not re.search(r"[A-Za-zÀ-ÿ]", txt):
        return None
    if txt.lower() in STOP or txt.isdigit():
        return None
    return txt


class Linguistique:
    """Accumulateur : tout ce qu'on extrait de spaCy en UN SEUL passage (v2 : 2 passes, NER tronqué)."""

    def __init__(self, repos_labels: list[str]):
        self.repo_idx = {r: i for i, r in enumerate(repos_labels)}
        self.vocab: dict[str, int] = {}
        self.ivocab: list[str] = []
        self.lemmas: list[list[str]] = []          # par passage
        self.n_tok: list[int] = []                  # tokens alphabétiques (stop inclus)
        self.n_sent: list[int] = []
        self.freq = Counter()                       # nb d'unités (phrases) contenant le nœud
        self.tok_count = Counter()                  # nb d'occurrences (lemmes de contenu)
        self.cooc = Counter()
        self.dist_sum = Counter()
        self.node_repos = defaultdict(set)
        self.n_units = 0
        self.bigrams = Counter()
        self.ner_count = Counter()
        self.ner_surface = defaultdict(Counter)
        self.ner_type = defaultdict(Counter)
        self.ner_repo = defaultdict(Counter)
        self.node_type: dict[int, str] = {}

    def _id(self, mot: str) -> int:
        i = self.vocab.get(mot)
        if i is None:
            i = self.vocab[mot] = len(self.ivocab)
            self.ivocab.append(mot)
        return i

    def ingest(self, repo: str, phrases: list[dict]):
        """phrases : [{'seq': [(lemme, idx)], 'nodes': [(cle, idx, prio)], 'ents': [(cle, label, surface)], 'ntok': n}]"""
        ri = self.repo_idx.get(repo, 0)
        lem_passage, ntok, nsent = [], 0, 0
        for ph in phrases:
            nsent += 1
            ntok += ph["ntok"]
            seq = ph["seq"]
            lem_passage += [l for l, _ in seq]
            ids_seq = [(self._id(l), i) for l, i in seq]
            for (a, ia), (b, ib) in zip(ids_seq, ids_seq[1:]):
                if ib == ia + 1 and a != b:
                    self.bigrams[(a, b)] += 1
            for a, _ in ids_seq:
                self.tok_count[a] += 1
            for cle, lab, surf in ph["ents"]:
                self.ner_count[cle] += 1
                self.ner_surface[cle][surf] += 1
                self.ner_type[cle][lab] += 1
                self.ner_repo[cle][repo] += 1
            # nœuds du graphe pour cette unité (entités et lexique d'abord, plafond 30)
            nodes = sorted(ph["nodes"], key=lambda x: (-x[2], x[1]))
            vus, unite = set(), []
            for cle, idx, prio in nodes:
                if cle in vus:
                    continue
                vus.add(cle)
                unite.append((self._id(cle), idx))
                if len(unite) >= 30:
                    break
            if len(unite) < 2:
                if unite:
                    self.n_units += 1
                    self.freq[unite[0][0]] += 1
                    self.node_repos[unite[0][0]].add(ri)
                continue
            self.n_units += 1
            for a, _ in unite:
                self.freq[a] += 1
                self.node_repos[a].add(ri)
            for (a, ia), (b, ib) in combinations(unite, 2):
                if a == b:
                    continue
                k = (a << 21 | b) if a < b else (b << 21 | a)
                self.cooc[k] += 1
                self.dist_sum[k] += abs(ia - ib)
            for cle, lab, _ in ph["ents"]:
                self.node_type[self._id(cle)] = lab
        self.lemmas.append(lem_passage)
        self.n_tok.append(ntok)
        self.n_sent.append(nsent)


# ── Extraction de phrases : version spaCy ─────────────────────────────────────

def _phrases_spacy(doc) -> list[dict]:
    phrases = []
    for sent in doc.sents:
        ents_span = [e for e in sent.ents if e.label_ in NER_LABELS]
        couverts = set()
        ents, nodes = [], []
        for e in ents_span:
            surf = _nettoyer_entite(e.text)
            if not surf:
                continue
            cle = surf.lower()
            ents.append((cle, e.label_, surf))
            nodes.append((cle, e.start, 3))
            couverts.update(range(e.start, e.end))
        seq, ntok = [], 0
        for t in sent:
            if t.is_space or t.is_punct or t.like_num or t.like_url:
                continue
            low = t.lower_
            if not (low in LEXIQUE_SET or re.fullmatch(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", t.text)):
                continue
            ntok += 1
            lem = low if low in LEXIQUE_SET else (t.lemma_ or low).lower().strip()
            if not re.fullmatch(r"[^\W\d]+(?:[\-'’_][^\W\d]+)*", lem or ""):
                lem = low
            if len(lem) < 3 or lem in STOP or low in STOP:
                continue
            seq.append((lem, t.i))
            if t.i in couverts:
                continue
            if lem in LEXIQUE_SET:
                nodes.append((lem, t.i, 2))
            elif t.pos_ in ("NOUN", "PROPN") and len(lem) > 4:
                nodes.append((lem, t.i, 1))
        phrases.append({"seq": seq, "nodes": nodes, "ents": ents, "ntok": ntok})
    return phrases


# ── Extraction de phrases : mode de secours (sans modèle spaCy) ───────────────

RE_PHRASE = re.compile(r"(?<=[.!?…])\s+|\n+")

def _phrases_regex(texte: str) -> list[dict]:
    phrases = []
    for s in RE_PHRASE.split(texte):
        if not s or not s.strip():
            continue
        seq, nodes, ents, ntok = [], [], [], 0
        mots = list(RE_WORD.finditer(s))
        i = 0
        while i < len(mots):
            m = mots[i]
            w = m.group(0).strip("-'’_")
            low = w.lower()
            ntok += 1
            majus = w[:1].isupper() and i > 0
            if majus and low not in STOP and len(w) > 2:  # entité heuristique (suite de Majuscules)
                j = i
                parts = [w]
                while j + 1 < len(mots) and mots[j + 1].group(0)[:1].isupper() and len(parts) < 4:
                    j += 1
                    parts.append(mots[j].group(0))
                surf = _nettoyer_entite(" ".join(parts))
                if surf:
                    ents.append((surf.lower(), "MISC", surf))
                    nodes.append((surf.lower(), i, 3))
                for k in range(i, j + 1):
                    ntok += 1 if k > i else 0
                    lk = mots[k].group(0).lower().strip("-'’_")
                    if len(lk) >= 3 and lk not in STOP:
                        seq.append((lk, k))
                i = j + 1
                continue
            if len(low) >= 3 and low not in STOP:
                seq.append((low, i))
                if low in LEXIQUE_SET:
                    nodes.append((low, i, 2))
                elif len(low) > 6:
                    nodes.append((low, i, 1))
            i += 1
        phrases.append({"seq": seq, "nodes": nodes, "ents": ents, "ntok": ntok})
    return phrases


def passe_linguistique(passages: list[dict], repos_labels: list[str]) -> Linguistique:
    section("Passe linguistique unique (lemmes • NER • graphe • collocations)", C.BCYAN, "◈")
    nlp = _charger_spacy()
    L = Linguistique(repos_labels)
    textes = [canon_text(p["texte"]) for p in passages]
    total = len(passages)
    t0 = time.time()
    if nlp is not None:
        stream = nlp.pipe(textes, batch_size=64, n_process=max(1, CFG.n_procs))
        for i, doc in enumerate(stream):
            L.ingest(passages[i]["repo"], _phrases_spacy(doc))
            if i % 200 == 0 or i == total - 1:
                progress_bar(i + 1, total, prefix="spaCy")
    else:
        for i, t in enumerate(textes):
            L.ingest(passages[i]["repo"], _phrases_regex(t))
            if i % 500 == 0 or i == total - 1:
                progress_bar(i + 1, total, prefix="regex")
    kv_line("Passages analysés", fmt_int(total))
    kv_line("Lemmes de contenu", fmt_int(sum(len(l) for l in L.lemmas)))
    kv_line("Vocabulaire (types)", fmt_int(len(L.ivocab)))
    kv_line("Unités (phrases) graphe", fmt_int(L.n_units))
    kv_line("Paires co-occurrentes", fmt_int(len(L.cooc)))
    kv_line("Entités distinctes", fmt_int(len(L.ner_count)))
    kv_line("Durée", f"{time.time() - t0:.1f}s")
    return L


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3b — TEXT MINING (stylométrie • TF-IDF • log-odds • collocations • NER •
#             LDA auto-k • embeddings • thèmes sémantiques • similarité entre dépôts)
# ══════════════════════════════════════════════════════════════════════════════

def _identity(x):
    return x


def lemmes_par(L: Linguistique, passages: list[dict], cle: str) -> dict:
    out = defaultdict(list)
    for p, lem in zip(passages, L.lemmas):
        out[p[cle]] += lem
    return out


# ── Stylométrie ───────────────────────────────────────────────────────────────

def mattr(tokens: list[str], w: int = 200) -> float:
    """Moving-Average TTR : diversité lexicale robuste à la longueur du texte."""
    tokens = tokens[:80_000]
    if len(tokens) < w:
        return len(set(tokens)) / max(len(tokens), 1)
    c = Counter(tokens[:w])
    s, n = len(c), 1
    for i in range(w, len(tokens)):
        out = tokens[i - w]
        c[out] -= 1
        if c[out] == 0:
            del c[out]
        c[tokens[i]] += 1
        s += len(c)
        n += 1
    return s / (n * w)


def pente_zipf(freqs: list[int]):
    f = np.array(sorted(freqs, reverse=True), float)
    hi = min(len(f), 1500)
    if hi < 30:
        return None
    r = np.arange(1, hi + 1)
    return float(np.polyfit(np.log(r[2:]), np.log(f[2:hi]), 1)[0])


def stylometrie(L, passages) -> pd.DataFrame:
    log("Stylométrie (diversité lexicale, Zipf, densité lexique Trame)...", "run")
    lem_repo = lemmes_par(L, passages, "repo")
    ntok, nsent, nfr, npass = Counter(), Counter(), Counter(), Counter()
    for p, nt, ns in zip(passages, L.n_tok, L.n_sent):
        ntok[p["repo"]] += nt
        nsent[p["repo"]] += ns
        npass[p["repo"]] += 1
        nfr[p["repo"]] += p["langue"] == "fr"
    rows = []
    for repo, lem in lem_repo.items():
        if not lem:
            continue
        c = Counter(lem)
        trame = sum(v for k, v in c.items() if k in LEXIQUE_SET)
        rows.append({
            "repo": repo, "passages": npass[repo], "tokens": ntok[repo], "lemmes": len(lem),
            "types": len(c), "ttr": round(len(c) / len(lem), 4), "mattr": round(mattr(lem), 4),
            "hapax_pct": round(100 * sum(1 for v in c.values() if v == 1) / len(c), 1),
            "zipf_pente": (round(pente_zipf(list(c.values())), 3) if pente_zipf(list(c.values())) else None),
            "mots_par_phrase": round(ntok[repo] / max(nsent[repo], 1), 1),
            "densite_trame_pm": round(1000 * trame / len(lem), 2),
            "pct_fr": round(100 * nfr[repo] / npass[repo], 1),
        })
    return pd.DataFrame(rows).sort_values("tokens", ascending=False).reset_index(drop=True)


# ── TF-IDF (documents) ────────────────────────────────────────────────────────

def tfidf_global_et_par_depot(L, passages, top_n=50):
    log("TF-IDF documentaire + termes par dépôt...", "run")
    lem_doc = lemmes_par(L, passages, "doc_id")
    docs = [l for l in lem_doc.values() if len(l) >= 5]
    res = {"global": [], "par_depot": {}}
    if len(docs) >= 5:
        vec = TfidfVectorizer(analyzer=_identity, min_df=2, max_df=0.7, sublinear_tf=True, max_features=8000)
        X = vec.fit_transform(docs)
        sc = np.asarray(X.sum(axis=0)).ravel()
        termes = vec.get_feature_names_out()
        res["global"] = [{"concept": pretty(termes[i]), "score": round(float(sc[i]), 4)}
                         for i in sc.argsort()[::-1][:top_n]]
    lem_repo = lemmes_par(L, passages, "repo")
    repos = [r for r, l in lem_repo.items() if len(l) >= 50]
    if len(repos) >= 3:
        vec = TfidfVectorizer(analyzer=_identity, min_df=1, max_df=0.6, sublinear_tf=True)
        X = vec.fit_transform([lem_repo[r] for r in repos])
        termes = vec.get_feature_names_out()
        for i, r in enumerate(repos):
            row = X[i].toarray().ravel()
            res["par_depot"][r] = [{"terme": pretty(termes[j]), "score": round(float(row[j]), 4)}
                                   for j in row.argsort()[::-1][:12] if row[j] > 0]
    return res


# ── Mots distinctifs : log-odds avec prior de Dirichlet informatif (Monroe et al. 2008) ──

def mots_distinctifs(L, passages, alpha0=500.0, top=15, min_tokens=300, vocab_max=15000):
    log("Termes distinctifs par dépôt (log-odds + prior informatif)...", "run")
    lem_repo = lemmes_par(L, passages, "repo")
    repos = [r for r, l in lem_repo.items() if len(l) >= min_tokens]
    if len(repos) < 2:
        return {}
    total = Counter()
    for r in repos:
        total.update(lem_repo[r])
    vocab = [w for w, n in total.most_common(vocab_max) if n >= 3]
    idx = {w: i for i, w in enumerate(vocab)}
    M = np.zeros((len(repos), len(vocab)))
    for i, r in enumerate(repos):
        for w, n in Counter(lem_repo[r]).items():
            j = idx.get(w)
            if j is not None:
                M[i, j] = n
    n_w = M.sum(0)
    prior = alpha0 * n_w / n_w.sum()
    out = {}
    for i, r in enumerate(repos):
        yi, yr = M[i], n_w - M[i]
        ni, nr = yi.sum(), yr.sum()
        d = (np.log((yi + prior) / (ni + alpha0 - yi - prior))
             - np.log((yr + prior) / (nr + alpha0 - yr - prior)))
        z = d / np.sqrt(1 / (yi + prior) + 1 / (yr + prior))
        ordre = [j for j in z.argsort()[::-1] if yi[j] >= 2][:top]
        out[r] = [{"terme": pretty(vocab[j]), "z": round(float(z[j]), 2), "n": int(yi[j])} for j in ordre]
    return out


# ── Collocations (NPMI sur adjacence réelle) ──────────────────────────────────

def collocations(L: Linguistique, top=40, min_count=3):
    log("Collocations (bigrammes à forte association NPMI)...", "run")
    N = max(sum(L.tok_count.values()), 1)
    rows = []
    for (a, b), c in L.bigrams.items():
        if c < min_count:
            continue
        pa, pb, pab = L.tok_count[a] / N, L.tok_count[b] / N, c / N
        npmi = math.log(pab / (pa * pb)) / (-math.log(pab))
        rows.append((f"{pretty(L.ivocab[a])} {pretty(L.ivocab[b])}", c, round(npmi, 3)))
    rows.sort(key=lambda x: -(x[2] * math.log1p(x[1])))
    return [{"bigramme": t, "n": c, "npmi": s} for t, c, s in rows[:top]]


# ── NER ───────────────────────────────────────────────────────────────────────

def synthese_ner(L: Linguistique, top=40):
    log("Synthèse NER (types, surfaces, ubiquité entre dépôts)...", "run")
    def surface(k):
        return L.ner_surface[k].most_common(1)[0][0]
    def typ(k):
        return L.ner_type[k].most_common(1)[0][0]
    top_k = [k for k, _ in L.ner_count.most_common(top)]
    par_type = defaultdict(list)
    for k, v in L.ner_count.most_common(600):
        par_type[typ(k)].append({"entite": surface(k), "frequence": v})
    return {
        "top_entites": [{"entite": surface(k), "frequence": L.ner_count[k], "type": typ(k),
                        "depots": len(L.ner_repo[k])} for k in top_k],
        "par_type": {t: v[:10] for t, v in par_type.items()},
        "ubiquistes": [{"entite": surface(k), "depots": len(L.ner_repo[k]), "frequence": L.ner_count[k]}
                       for k in sorted(L.ner_count, key=lambda k: (-len(L.ner_repo[k]), -L.ner_count[k]))[:15]],
        "matrice_repo": {surface(k): dict(L.ner_repo[k].most_common(8)) for k in top_k[:25]},
    }


# ── LDA avec choix automatique de k (cohérence NPMI) ──────────────────────────

def coherence_npmi(mots_par_topic: list[list[int]], Xbin) -> float:
    D = Xbin.shape[0]
    df = np.asarray(Xbin.sum(0)).ravel()
    co = (Xbin.T @ Xbin).tocsr()
    scores = []
    for mots in mots_par_topic:
        s = []
        for i, j in combinations(mots, 2):
            pij = co[i, j] / D
            if pij <= 0:
                s.append(-1.0)
                continue
            pi, pj = df[i] / D, df[j] / D
            s.append(math.log(pij / (pi * pj)) / (-math.log(pij)))
        scores.append(np.mean(s) if s else 0.0)
    return float(np.mean(scores))


def topics_lda(L, passages, n_mots=10):
    log("LDA : k automatique par cohérence NPMI...", "run")
    keep = [i for i, l in enumerate(L.lemmas) if len(l) >= 8]
    if len(keep) < 40:
        return [], None, []
    docs = [L.lemmas[i] for i in keep]
    vec = CountVectorizer(analyzer=_identity, min_df=3, max_df=0.5, max_features=4000)
    X = vec.fit_transform(docs)
    termes = vec.get_feature_names_out()
    Xbin = (X > 0).astype(np.float32).tocsr()
    rng = np.random.RandomState(CFG.seed)
    sub = rng.choice(X.shape[0], min(X.shape[0], 6000), replace=False)
    candidats = [CFG.n_topics] if CFG.n_topics else [4, 6, 8, 10, 12]
    candidats = [k for k in candidats if k * 5 <= X.shape[0]] or [4]
    meilleur, essais = None, []
    for k in candidats:
        lda = LatentDirichletAllocation(n_components=k, learning_method="online", max_iter=15,
                                        random_state=CFG.seed, batch_size=512)
        lda.fit(X[sub])
        tops = [list(c.argsort()[::-1][:n_mots]) for c in lda.components_]
        coh = coherence_npmi(tops, Xbin[sub])
        essais.append((k, round(coh, 4)))
        if meilleur is None or coh > meilleur[1]:
            meilleur = (k, coh)
    k = meilleur[0]
    kv_line("LDA — k testés (cohérence)", "  ".join(f"k={a}:{b}" for a, b in essais))
    lda = LatentDirichletAllocation(n_components=k, learning_method="online", max_iter=20,
                                    random_state=CFG.seed, batch_size=512)
    theta = lda.fit_transform(X)
    topics = []
    for t, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[::-1][:n_mots]
        topics.append({
            "topic_id": t, "mots": [pretty(termes[j]) for j in top_idx],
            "poids": [round(float(comp[j]), 4) for j in top_idx],
            "label": " · ".join(pretty(termes[j]) for j in top_idx[:3]),
            "prevalence": round(float(theta[:, t].mean()), 4),
        })
    # topic × dépôt
    rep = np.array([passages[i]["repo"] for i in keep])
    repos = sorted(set(rep))
    mat = np.array([theta[rep == r].mean(0) for r in repos])
    info = {"repos": repos, "matrice": mat, "essais": essais, "k": k}
    return topics, info, essais


# ── Embeddings, thèmes sémantiques, carte 2D ──────────────────────────────────

def _selection_stratifiee(passages, n_max, seed):
    cand = [i for i, p in enumerate(passages) if len(p["texte"].split()) >= 15]
    if len(cand) <= n_max:
        return cand
    rng = random.Random(seed)
    par = defaultdict(list)
    for i in cand:
        par[passages[i]["repo"]].append(i)
    ratio = n_max / len(cand)
    sel = []
    for lst in par.values():
        sel += rng.sample(lst, min(len(lst), max(int(len(lst) * ratio), 40)))
    return sorted(sel)[:int(n_max * 1.15)]


def representation_semantique(passages, L):
    """Embeddings multilingues (sentence-transformers) ; sinon repli TF-IDF + SVD."""
    sel = _selection_stratifiee(passages, CFG.max_embed, CFG.seed)
    if len(sel) < 60:
        return None
    modele = None
    if CFG.embeddings and HAS_ST:
        h = hashlib.sha1((CFG.embed_model + "".join(passages[i]["texte"][:80] for i in sel)).encode()).hexdigest()[:16]
        cache = CFG.cache_dir / f"emb_{h}.npy"
        try:
            modele = sentence_transformers.SentenceTransformer(CFG.embed_model)
            if cache.exists():
                emb = np.load(cache)
                log(f"Embeddings en cache ({emb.shape})", "ok")
            else:
                log(f"Embeddings {CFG.embed_model} sur {len(sel)} passages (CPU : quelques minutes)...", "run")
                emb = modele.encode([passages[i]["texte"][:1500] for i in sel], batch_size=64,
                                    show_progress_bar=False, normalize_embeddings=True)
                np.save(cache, emb)
            return {"sel": sel, "emb": np.asarray(emb), "modele": modele, "methode": CFG.embed_model}
        except Exception as e:
            log(f"Embeddings indisponibles ({str(e)[:90]}) → repli TF-IDF/SVD", "warn")
            modele = None
    docs = [L.lemmas[i] for i in sel]
    X = TfidfVectorizer(analyzer=_identity, min_df=3, max_df=0.5, sublinear_tf=True,
                        max_features=6000).fit_transform(docs)
    n_comp = min(80, X.shape[1] - 1, X.shape[0] - 1)
    Z = TruncatedSVD(n_components=n_comp, random_state=CFG.seed).fit_transform(X)
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    return {"sel": sel, "emb": Z, "modele": None, "methode": "TF-IDF + SVD (repli)"}


def mmr(doc_emb, cand_emb, k, diversity=0.6):
    sims = cand_emb @ doc_emb
    cc = cand_emb @ cand_emb.T
    sel = [int(np.argmax(sims))]
    rest = set(range(len(sims))) - set(sel)
    while len(sel) < k and rest:
        r = list(rest)
        mx = cc[np.ix_(r, sel)].max(1)
        j = r[int(np.argmax((1 - diversity) * sims[r] - diversity * mx))]
        sel.append(j)
        rest.discard(j)
    return sel


def ctfidf(class_counters: dict, top=10, vocab_max=15000):
    tot = Counter()
    for c in class_counters.values():
        tot.update(c)
    vocab = [w for w, n in tot.most_common(vocab_max) if n >= 5]
    if not vocab:
        return {k: [] for k in class_counters}
    classes = sorted(class_counters)
    M = np.array([[class_counters[c].get(w, 0) for w in vocab] for c in classes], float)
    tf = M / np.maximum(M.sum(1, keepdims=True), 1)
    idf = np.log(1 + M.sum(1).mean() / np.maximum(M.sum(0), 1))
    S = tf * idf
    return {c: [pretty(vocab[j]) for j in S[i].argsort()[::-1][:top]] for i, c in enumerate(classes)}


def projeter_2d(emb):
    n = len(emb)
    try:
        if HAS_UMAP:
            return umap_mod.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                                 random_state=CFG.seed).fit_transform(emb)
        if n <= 5000:
            from sklearn.manifold import TSNE
            return TSNE(n_components=2, perplexity=min(30, n // 4), init="pca",
                        random_state=CFG.seed, metric="cosine").fit_transform(emb)
    except Exception as e:
        log(f"Projection avancée en échec ({str(e)[:60]}) → SVD", "warn")
    return TruncatedSVD(n_components=2, random_state=CFG.seed).fit_transform(emb)


def themes_semantiques(rep, passages, L, k_force=0):
    log("Thèmes sémantiques (KMeans sur embeddings + étiquettes c-TF-IDF / MMR)...", "run")
    emb, sel = rep["emb"], rep["sel"]
    n = len(emb)
    ks = [k_force] if k_force else list(range(6, 15))
    ks = [k for k in ks if k < n // 10] or [min(6, max(2, n // 20))]
    ech = np.random.RandomState(CFG.seed).choice(n, min(n, 2500), replace=False)
    meil, best = None, -2
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=3, random_state=CFG.seed).fit_predict(emb)
        try:
            s = silhouette_score(emb[ech], lab[ech]) if len(set(lab[ech])) > 1 else -1
        except Exception:
            s = -1
        if s > best:
            meil, best = (k, lab), s
    k, lab = meil
    km = KMeans(n_clusters=k, n_init=5, random_state=CFG.seed).fit(emb)
    lab = km.labels_
    kv_line("Thèmes retenus (silhouette)", f"k={k}  ({best:.3f})  via {rep['methode']}")
    cls = {c: Counter() for c in range(k)}
    for pos, c in enumerate(lab):
        cls[c].update(L.lemmas[sel[pos]])
    labels_ct = ctfidf(cls, top=10)

    # Étiquettes sémantiques MMR (KeyBERT-style SANS troncature : on compare aux centroïdes)
    mmr_labels, glob_kw = {c: [] for c in range(k)}, []
    modele = rep.get("modele")
    if modele is not None:
        N = max(sum(L.tok_count.values()), 1)
        cands = [L.ivocab[i] for i, n_ in L.tok_count.most_common(900)
                 if len(L.ivocab[i]) >= 4 and n_ >= 3]
        phr = [t["bigramme"] for t in collocations(L, top=150, min_count=3)]
        cands = list(dict.fromkeys([pretty(c) for c in cands] + phr))
        try:
            cemb = modele.encode(cands, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
            centre = emb.mean(0)
            centre /= np.linalg.norm(centre) + 1e-9
            idx = mmr(centre, cemb, 50, diversity=0.6)
            glob_kw = [{"concept": cands[i], "score": round(float(cemb[i] @ centre), 4)} for i in idx]
            for c in range(k):
                ctr = km.cluster_centers_[c] / (np.linalg.norm(km.cluster_centers_[c]) + 1e-9)
                mmr_labels[c] = [cands[i] for i in mmr(ctr, cemb, 8, diversity=0.5)]
        except Exception as e:
            log(f"MMR sémantique en échec : {str(e)[:80]}", "warn")

    themes = []
    for c in range(k):
        membres = np.where(lab == c)[0]
        d = np.linalg.norm(emb[membres] - km.cluster_centers_[c], axis=1)
        proto = passages[sel[membres[int(d.argmin())]]]
        repos = Counter(passages[sel[m]]["repo"] for m in membres)
        themes.append({
            "theme_id": c, "taille": int(len(membres)),
            "mots_cles_ctfidf": labels_ct[c], "mots_cles_semantiques": mmr_labels[c],
            "label": " · ".join((mmr_labels[c][:3] or labels_ct[c][:3])),
            "depots_dominants": [{"depot": r, "passages": v} for r, v in repos.most_common(4)],
            "passage_prototype": {"depot": proto["repo"], "fichier": proto["path"],
                                  "extrait": re.sub(r"\s+", " ", proto["texte"])[:260]},
        })
    xy = projeter_2d(emb)
    repos_all = sorted({p["repo"] for p in passages})
    comp = np.zeros((len(repos_all), k))
    ri = {r: i for i, r in enumerate(repos_all)}
    for pos, c in enumerate(lab):
        comp[ri[passages[sel[pos]]["repo"]], c] += 1
    return {"themes": themes, "labels": lab, "xy": xy, "k": k, "silhouette": float(best), "sel": list(sel),
            "mots_cles_globaux": glob_kw, "repos": repos_all, "composition": comp}


# ── Similarité entre dépôts (sémantique + doublons exacts) ────────────────────

def similarite_depots(L, passages, cor: Corpus):
    log("Similarité entre dépôts (TF-IDF cosinus, clustering hiérarchique, forks)...", "run")
    lem_repo = lemmes_par(L, passages, "repo")
    repos = [r for r, l in lem_repo.items() if len(l) >= 50]
    if len(repos) < 3:
        return None
    X = TfidfVectorizer(analyzer=_identity, min_df=1, max_df=0.8, sublinear_tf=True) \
        .fit_transform([lem_repo[r] for r in repos])
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 1.0)
    ordre = list(sch.leaves_list(sch.linkage(pdist(sim), "average")))
    hs = [cor.hash_par_repo.get(r, set()) for r in repos]
    jac = np.zeros_like(sim)
    for i, j in combinations(range(len(repos)), 2):
        u = len(hs[i] | hs[j])
        jac[i, j] = jac[j, i] = (len(hs[i] & hs[j]) / u) if u else 0
    paires = []
    for i, j in combinations(range(len(repos)), 2):
        if sim[i, j] >= 0.6 or jac[i, j] >= 0.2:
            paires.append({"a": repos[i], "b": repos[j], "cosinus": round(float(sim[i, j]), 3),
                           "fichiers_identiques_jaccard": round(float(jac[i, j]), 3)})
    paires.sort(key=lambda x: -x["cosinus"])
    return {"repos": repos, "sim": sim, "ordre": ordre, "jaccard_hash": jac, "paires": paires[:30]}


# ── Commits ───────────────────────────────────────────────────────────────────

def analyser_commits(cdf: pd.DataFrame):
    if cdf is None or len(cdf) < 3:
        return None
    log("Temporalité : commits par mois, rythme hebdo, auteurs, vocabulaire des messages...", "run")
    d = cdf.copy()
    d["mois"] = d["date"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    d["jour"] = d["date"].dt.dayofweek
    d["heure"] = d["date"].dt.hour
    mensuel = d.groupby(["mois", "repo"]).size().unstack(fill_value=0)
    rythme = np.zeros((7, 24))
    for j, h in zip(d["jour"], d["heure"]):
        rythme[j, h] += 1
    mots = Counter()
    for s in d["sujet"]:
        for w in RE_WORD.findall(s.lower()):
            w = w.strip("-'’_")
            if len(w) >= 4 and w not in STOP and w not in {"update", "merge", "commit", "initial", "added", "fixed"}:
                mots[w] += 1
    par_repo = d.groupby("repo").agg(commits=("sha", "count"), debut=("date", "min"), fin=("date", "max"),
                                     auteurs=("auteur", "nunique")).reset_index()
    par_repo["duree_jours"] = (par_repo["fin"] - par_repo["debut"]).dt.days
    return {"mensuel": mensuel, "rythme": rythme, "mots_messages": mots.most_common(40),
            "par_repo": par_repo, "auteurs": d["auteur"].value_counts().head(10).to_dict(),
            "n": int(len(d))}


# ── Orchestrateur Phase III ───────────────────────────────────────────────────

def analyser_le_noyau_non_lineaire(cor: Corpus, repos: list[RepoInfo], cdf: pd.DataFrame):
    banner("PHASE III — ANALYSE DU NOYAU NON-LINÉAIRE", "Text mining multi-méthodes",
           couleur=C.BBLUE, symbole="❖")
    passages = decouper_passages(cor.docs)
    kv_line("Passages (unité d'analyse)", fmt_int(len(passages)))
    labels = [r.label for r in repos]
    L = passe_linguistique(passages, labels)

    section("Extraction des concepts (multi-méthodes)", C.BCYAN, "◈")
    A = {"passages": passages, "L": L}
    A["style"] = safe("stylométrie", stylometrie, L, passages, default=pd.DataFrame())
    A["tfidf_full"] = safe("tfidf", tfidf_global_et_par_depot, L, passages, default={"global": [], "par_depot": {}})
    A["distinctifs"] = safe("log-odds", mots_distinctifs, L, passages, default={})
    A["collocations"] = safe("collocations", collocations, L, default=[])
    A["ner"] = safe("ner", synthese_ner, L, default={"top_entites": [], "par_type": {}})
    topics, lda_info, _ = safe("lda", topics_lda, L, passages, default=([], None, []))
    A["topics"], A["lda_info"] = topics, lda_info

    section("Sémantique (embeddings, thèmes, carte)", C.BMAGENTA, "◈")
    rep = safe("embeddings", representation_semantique, passages, L, default=None)
    A["sem"] = safe("thèmes", themes_semantiques, rep, passages, L, default=None) if rep else None
    A["similarite"] = safe("similarité", similarite_depots, L, passages, cor, default=None)
    A["commits"] = safe("commits", analyser_commits, cdf, default=None)

    kb = (A["sem"] or {}).get("mots_cles_globaux", [])
    tf = A["tfidf_full"]["global"]
    concepts = {}
    for k in kb:
        concepts.setdefault(k["concept"], {})["keybert"] = k["score"]
    for k in tf:
        concepts.setdefault(k["concept"], {})["tfidf"] = k["score"]
    A.update({"keybert": kb, "tfidf": tf, "concepts_fusionnes": concepts})

    # Aperçu console
    section("Aperçu — mots distinctifs (z-score) par dépôt", C.BGREEN, "★")
    for r, lst in list(A["distinctifs"].items())[:8]:
        print(f"    {C.BWHITE}{r[:34]:<34}{C.RESET} " + ", ".join(t["terme"] for t in lst[:6]))
    if A["sem"]:
        section("Aperçu — thèmes sémantiques", C.BYELLOW, "★")
        for t in sorted(A["sem"]["themes"], key=lambda x: -x["taille"])[:8]:
            print(f"    {C.BYELLOW}◉{C.RESET} {t['taille']:>5} pass.  {t['label']}")
    if A["collocations"]:
        section("Aperçu — collocations", C.BCYAN, "★")
        print("    " + " │ ".join(c["bigramme"] for c in A["collocations"][:8]))
    return A, L


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3c — GRAPHE ONTOLOGIQUE (co-occurrence multi-pondérée, NPMI corrigé)
# ══════════════════════════════════════════════════════════════════════════════

def construire_graphe(L: Linguistique) -> nx.Graph:
    """Arêtes : weight (co-occurrences), npmi, distance (tokens), cost (=1/weight, pour les plus courts chemins).
    v2 : PMI calculé avec N = Σ freq (≠ nombre d'unités) => NPMI biaisé ; 'distance' servait de coût
    pour la betweenness (non pertinent) ; listes de distances stockées pour chaque paire (mémoire)."""
    log("Tissage du graphe de co-occurrence multi-pondéré...", "run")
    G = nx.Graph()
    N = max(L.n_units, 1)
    masque = (1 << 21) - 1
    for key, c in L.cooc.items():
        if c < CFG.min_cooc:
            continue
        a, b = key >> 21, key & masque
        pa, pb, pab = L.freq[a] / N, L.freq[b] / N, c / N
        if pab >= 1 or pa <= 0 or pb <= 0:
            continue
        npmi = math.log(pab / (pa * pb)) / (-math.log(pab))
        G.add_edge(L.ivocab[a], L.ivocab[b], weight=int(c), npmi=round(float(npmi), 4),
                   distance=round(L.dist_sum[key] / c, 2), cost=1.0 / c)
    for n in G.nodes():
        i = L.vocab[n]
        G.nodes[n]["freq"] = int(L.freq[i])
        G.nodes[n]["n_repos"] = len(L.node_repos[i])
        G.nodes[n]["trame"] = int(n in LEXIQUE_SET)
        G.nodes[n]["ner_type"] = L.node_type.get(i, "")
        G.nodes[n]["label"] = pretty(n)
    log(f"Graphe construit : {G.number_of_nodes()} nœuds / {G.number_of_edges()} arêtes", "ok")
    return G


def _diametre_approx(G: nx.Graph, essais=4):
    """Borne inférieure par double balayage BFS (nx.diameter exact = O(n·m) : gèle sur 10k+ nœuds)."""
    if G.number_of_nodes() < 2:
        return 0
    start = max(G.degree, key=lambda x: x[1])[0]
    best = 0
    for _ in range(essais):
        d = nx.single_source_shortest_path_length(G, start)
        start, ecc = max(d.items(), key=lambda x: x[1])
        best = max(best, ecc)
    return best


def _communautes(G: nx.Graph):
    """Louvain (networkx ≥ 2.8 en natif ; python-louvain si présent) → partition triée par taille décroissante."""
    methode, comms = None, None
    try:
        comms = [set(c) for c in nx.community.louvain_communities(G, weight="weight", seed=CFG.seed)]
        methode = "Louvain (networkx)"
    except Exception:
        if HAS_LOUVAIN:
            try:
                p = louvain_mod.best_partition(G, weight="weight", random_state=CFG.seed)
                tmp = defaultdict(set)
                for n, c in p.items():
                    tmp[c].add(n)
                comms, methode = list(tmp.values()), "Louvain (python-louvain)"
            except Exception:
                pass
    if comms is None:
        try:
            comms = [set(c) for c in nx.community.greedy_modularity_communities(G, weight="weight")]
            methode = "Greedy Modularity"
        except Exception:
            comms, methode = [set(G.nodes())], "N/A"
    comms.sort(key=len, reverse=True)
    partition = {n: i for i, c in enumerate(comms) for n in c}
    try:
        mod = nx.community.modularity(G, comms, weight="weight")
    except Exception:
        mod = None
    return partition, mod, methode


def analyser_graphe(G: nx.Graph):
    """Centralités multiples + communautés nommées + coreness + participation + métriques globales."""
    section("Analyse structurelle du graphe", C.BCYAN, "◈")
    if G.number_of_edges() < 2:
        log("Graphe trop pauvre pour l'analyse structurelle.", "warn")
        return {}, G

    composantes = sorted(nx.connected_components(G), key=len, reverse=True)
    gc = composantes[0]
    G_gc = G.subgraph(gc).copy()
    n_cl = list(G.nodes())
    if len(n_cl) > 4000:
        n_cl = random.Random(CFG.seed).sample(n_cl, 4000)
    try:
        assort = round(nx.degree_pearson_correlation_coefficient(G), 4)
    except Exception:
        assort = "n/a"
    metriques = {
        "noeuds": G.number_of_nodes(), "aretes": G.number_of_edges(),
        "densite": round(nx.density(G), 5), "composantes": len(composantes),
        "composante_max": len(gc), "diametre_gc_approx": _diametre_approx(G_gc),
        "clustering_moy": round(nx.average_clustering(G, nodes=n_cl), 4),
        "transitivite": round(nx.transitivity(G), 4),
        "assortativite_degre": assort,
        "degre_moyen": round(2 * G.number_of_edges() / max(G.number_of_nodes(), 1), 2),
    }
    G_nl = G.copy()
    G_nl.remove_edges_from(list(nx.selfloop_edges(G_nl)))
    core = nx.core_number(G_nl)
    metriques["k_core_max"] = max(core.values()) if core else 0
    for k, v in metriques.items():
        kv_line(k.replace("_", " ").capitalize(), str(v))

    log("Centralités (pagerank, betweenness échantillonnée, eigenvector, k-core)...", "run")
    try:
        pagerank = nx.pagerank(G_gc, weight="weight", max_iter=200)
    except Exception:
        pagerank = {n: 1 / len(G_gc) for n in G_gc}
    degree_c = dict(G.degree(weight="weight"))
    try:
        betweenness = nx.betweenness_centrality(G_gc, weight="cost", normalized=True,
                                                k=min(300, len(G_gc)), seed=CFG.seed)
    except Exception:
        betweenness = {n: 0 for n in G_gc}
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G_gc, weight="weight")
    except Exception:
        try:
            eigenvector = nx.eigenvector_centrality(G_gc, max_iter=500, weight="weight")
        except Exception:
            eigenvector = {n: 0 for n in G_gc}

    partition, modularite, methode = _communautes(G)
    kv_line("Communautés", f"{len(set(partition.values()))} ({methode})")
    kv_line("Modularité", f"{modularite:.4f}" if modularite is not None else "n/a")

    # Coefficient de participation : un nœud "pont" relie plusieurs communautés
    kic = defaultdict(lambda: defaultdict(float))
    for u, v, d in G_nl.edges(data=True):
        w = d.get("weight", 1)
        kic[u][partition[v]] += w
        kic[v][partition[u]] += w
    participation = {}
    for n, dct in kic.items():
        tot = sum(dct.values())
        participation[n] = round(1 - sum((x / tot) ** 2 for x in dct.values()), 4) if tot else 0.0

    # Communautés : étiquettes (top PageRank/degré) + matrice inter-communautés
    comm_nodes = defaultdict(list)
    for n, c in partition.items():
        comm_nodes[c].append(n)
    score = lambda n: pagerank.get(n, 0) + 1e-9 * degree_c.get(n, 0)
    etiquettes = {c: " · ".join(pretty(n) for n in sorted(ns, key=lambda n: -score(n))[:3])
                  for c, ns in comm_nodes.items()}
    top_c = [c for c, _ in sorted(comm_nodes.items(), key=lambda x: -len(x[1]))[:12]]
    ic = {c: i for i, c in enumerate(top_c)}
    mat = np.zeros((len(top_c), len(top_c)))
    for u, v, d in G_nl.edges(data=True):
        a, b = partition[u], partition[v]
        if a in ic and b in ic:
            mat[ic[a], ic[b]] += d["weight"]
            if a != b:
                mat[ic[b], ic[a]] += d["weight"]

    section("Top 15 — Nœuds les plus centraux (PageRank)", C.BMAGENTA, "★")
    top_pr = sorted(pagerank.items(), key=lambda x: -x[1])[:15]
    mx = top_pr[0][1]
    for i, (n, s) in enumerate(top_pr, 1):
        print(f"    {C.BWHITE}{i:>2}.{C.RESET} {pretty(n):<25} {C.DIM}PR={s:.4f}{C.RESET}  "
              f"{C.BMAGENTA}{'█' * int(s / mx * 30)}{C.RESET}  {C.BCYAN}⌘{partition.get(n, '?')}{C.RESET}"
              f"  {C.DIM}k-core {core.get(n, 0)} • {G.nodes[n]['n_repos']} dépôts{C.RESET}")
    section("Top 10 — Ponts (betweenness) & nœuds-frontières (participation)", C.BYELLOW, "⌘")
    for i, (n, s) in enumerate(sorted(betweenness.items(), key=lambda x: -x[1])[:10], 1):
        print(f"    {C.BWHITE}{i:>2}.{C.RESET} {pretty(n):<30} {C.BYELLOW}{s:.4f}{C.RESET}"
              f"  {C.DIM}participation {participation.get(n, 0):.2f}{C.RESET}")
    section("Top 10 — Influence (eigenvector)", C.BGREEN, "♛")
    for i, (n, s) in enumerate(sorted(eigenvector.items(), key=lambda x: -x[1])[:10], 1):
        print(f"    {C.BWHITE}{i:>2}.{C.RESET} {pretty(n):<30} {C.BGREEN}{s:.4f}{C.RESET}")
    section("Composition des communautés", C.BCYAN, "▤")
    pal = [C.BCYAN, C.BMAGENTA, C.BGREEN, C.BYELLOW, C.BRED, C.BBLUE, C.BWHITE]
    for c in top_c[:8]:
        print(f"    {pal[c % len(pal)]}◉ {c}{C.RESET} ({len(comm_nodes[c])} nœuds) : {etiquettes[c]}")

    for n in G.nodes():  # types natifs (GraphML refuse numpy.int64)
        G.nodes[n]["pagerank"] = float(pagerank.get(n, 0))
        G.nodes[n]["betweenness"] = float(betweenness.get(n, 0))
        G.nodes[n]["eigenvector"] = float(eigenvector.get(n, 0))
        G.nodes[n]["degree_w"] = float(degree_c.get(n, 0))
        G.nodes[n]["community"] = int(partition.get(n, -1))
        G.nodes[n]["coreness"] = int(core.get(n, 0))
        G.nodes[n]["participation"] = float(participation.get(n, 0))
    return {
        "metriques": metriques, "pagerank": pagerank, "betweenness": betweenness,
        "eigenvector": eigenvector, "degree": degree_c, "partition": partition,
        "modularite": modularite, "methode_comm": methode, "coreness": core,
        "participation": participation, "etiquettes_comm": etiquettes,
        "communautes_top": top_c, "matrice_comm": mat,
    }, G


def noyau_visuel(G, ag, n_max=None, k_voisins=3):
    """Sous-graphe lisible : top-N PageRank, backbone = arbre couvrant maximal ∪ top-k voisins/nœud."""
    n_max = n_max or CFG.graph_viz_nodes
    pr = ag.get("pagerank", {})
    top = [n for n, _ in sorted(pr.items(), key=lambda x: -x[1])[:n_max] if n in G]
    H = G.subgraph(top).copy()
    if H.number_of_edges() == 0:
        return H
    garder = set()
    for n in H:
        voisins = sorted(H[n].items(), key=lambda kv: -(kv[1]["npmi"] * math.log1p(kv[1]["weight"])))
        garder.update(tuple(sorted((n, v))) for v, _ in voisins[:k_voisins])
    for u, v in nx.maximum_spanning_tree(H, weight="weight").edges():
        garder.add(tuple(sorted((u, v))))
    B = nx.Graph()
    B.add_nodes_from(H.nodes(data=True))
    for u, v in garder:
        B.add_edge(u, v, **H[u][v])
    return B


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — FORGE DU GRIMOIRE (JSON compatible v2 + CSV + GraphML + résumé LLM)
# ══════════════════════════════════════════════════════════════════════════════

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def _csv(df, nom):
    try:
        d = CFG.output_dir / "tables"
        d.mkdir(exist_ok=True)
        df.to_csv(d / nom, index=False, encoding="utf-8-sig")
    except Exception as e:
        log(f"CSV {nom} : {str(e)[:60]}", "warn")


def exporter_tables(A, ag, G, cor, cdf):
    L = A["L"]
    pr, part = ag.get("pagerank", {}), ag.get("partition", {})
    if G is not None and G.number_of_nodes():
        _csv(pd.DataFrame([{"noeud": n, **{k: v for k, v in d.items()}} for n, d in G.nodes(data=True)])
             .sort_values("pagerank", ascending=False), "noeuds.csv")
        ar = sorted(((u, v, d["weight"], d["npmi"], d["distance"]) for u, v, d in G.edges(data=True)),
                    key=lambda x: -x[2])[:60000]
        _csv(pd.DataFrame(ar, columns=["source", "cible", "weight", "npmi", "distance_moy"]), "aretes.csv")
    if len(A.get("style", [])):
        _csv(A["style"], "stylometrie.csv")
    rows = [{"depot": r, **t} for r, lst in A.get("distinctifs", {}).items() for t in lst]
    if rows:
        _csv(pd.DataFrame(rows), "mots_distinctifs.csv")
    if A.get("collocations"):
        _csv(pd.DataFrame(A["collocations"]), "collocations.csv")
    if A["ner"].get("top_entites"):
        _csv(pd.DataFrame(A["ner"]["top_entites"]), "entites.csv")
    if A.get("sem"):
        _csv(pd.DataFrame([{k: (", ".join(map(str, v)) if isinstance(v, list) and k.startswith("mots")
                               else v) for k, v in t.items() if k not in ("passage_prototype", "depots_dominants")}
                           for t in A["sem"]["themes"]]), "themes.csv")
    if cdf is not None and len(cdf):
        _csv(cdf.assign(date=cdf["date"].astype(str)), "commits.csv")
    _csv(cor.inventaire.groupby(["repo", "ext"], as_index=False).agg(fichiers=("octets", "size"),
                                                                       octets=("octets", "sum")), "inventaire.csv")
    if len(cor.references):
        _csv(cor.references, "references_inter_depots.csv")
    if len(cor.doublons):
        _csv(cor.doublons, "doublons.csv")
    if len(cor.code):
        _csv(cor.code, "code_python.csv")
    if len(cor.todos):
        _csv(cor.todos, "todos.csv")
    if A.get("similarite"):
        s = A["similarite"]
        _csv(pd.DataFrame(s["sim"], index=s["repos"], columns=s["repos"]).reset_index(names="depot"),
             "similarite_depots.csv")


def forger_le_grimoire(A, ag, G, cor, repos, cdf):
    banner("PHASE IV — FORGE DU GRIMOIRE", "Assemblage JSON / CSV / GraphML / Résumé LLM",
           couleur=C.BYELLOW, symbole="❖")
    pagerank, partition = ag.get("pagerank", {}), ag.get("partition", {})
    top_gods = sorted(pagerank.items(), key=lambda x: -x[1])[:20]
    ar_fortes = sorted(({"source": u, "cible": v, "force": d["weight"], "npmi": d["npmi"],
                         "distance": d["distance"]} for u, v, d in G.edges(data=True) if d["weight"] >= 3),
                       key=lambda x: (-x["force"], -x["npmi"])) if G.number_of_edges() else []
    top_nodes = sorted(G.nodes(), key=lambda n: -G.nodes[n].get("pagerank", 0))[:1500]
    membres = {str(c): sorted([n for n, cc in partition.items() if cc == c],
                              key=lambda x: -pagerank.get(x, 0))[:30] for c in set(partition.values())}
    sem = A.get("sem") or {}
    cm = A.get("commits")
    code_agg = (cor.code.groupby("repo").agg(fichiers=("path", "count"), loc=("loc", "sum"),
                                              fonctions=("fonctions", "sum"), classes=("classes", "sum"),
                                              complexite=("complexite", "sum"),
                                              doc_cov=("doc_cov", "mean")).round(3).reset_index().to_dict("records")
                if len(cor.code) else [])
    grimoire = {
        "METADONNEES": {
            "version": "3.0-ultra", "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "Extraction automatisée des strates GitHub MTT-2075",
            "principe_fondateur": ("La réalité est une solution triviale d'un système non-linéaire. "
                                   "Les Dieux sont les solutions idéales dans la mousse quantique."),
            "statistiques_moisson": cor.stats,
            "depots": [{"label": r.label, "url": r.url, "head": r.head[:10], "shallow": r.shallow} for r in repos],
            "parametres": {k: str(v) for k, v in vars(CFG).items() if k not in ("repos",)},
        },
        "LIBER_PRIMUS_VRAIS_NOMS": {"keybert_mmr": A.get("keybert", []), "tfidf": A.get("tfidf", []),
                                    "tfidf_par_depot": A["tfidf_full"].get("par_depot", {})},
        "LIBER_SECUNDUS_ENTITES": A.get("ner", {}),
        "LIBER_TERTIUS_TOPICS": A.get("topics", []),
        "LIBER_QUARTUS_SOLUTIONS_IDEALES": [{
            "entite": pretty(e), "pagerank": round(pagerank.get(e, 0), 5),
            "betweenness": round(ag["betweenness"].get(e, 0), 5),
            "eigenvector": round(ag["eigenvector"].get(e, 0), 5),
            "degree_pondere": ag["degree"].get(e, 0), "communaute": partition.get(e, -1),
            "k_core": ag["coreness"].get(e, 0), "participation": ag["participation"].get(e, 0),
            "depots": G.nodes[e]["n_repos"], "freq": G.nodes[e]["freq"],
        } for e, _ in top_gods],
        "LIBER_QUINTUS_COMMUNAUTES": {"methode": ag.get("methode_comm", "n/a"), "modularite": ag.get("modularite"),
                                      "etiquettes": {str(c): v for c, v in ag.get("etiquettes_comm", {}).items()
                                                     if c in ag.get("communautes_top", [])},
                                      "membres": membres},
        "LIBER_SEXTUS_THEMES_SEMANTIQUES": sem.get("themes", []),
        "LIBER_SEPTIMUS_TERMES_DISTINCTIFS": A.get("distinctifs", {}),
        "LIBER_OCTAVUS_COLLOCATIONS": A.get("collocations", []),
        "LIBER_NONUS_TEMPORALITE": ({"commits": cm["n"], "par_depot": cm["par_repo"].astype(str).to_dict("records"),
                                     "mots_messages": cm["mots_messages"], "auteurs": cm["auteurs"]} if cm else {}),
        "APPENDIX_METRIQUES": ag.get("metriques", {}),
        "APPENDIX_RELATIONNEL": {
            "noeuds": [{"id": n, **{k: G.nodes[n].get(k, 0) for k in
                                    ("pagerank", "betweenness", "eigenvector", "degree_w", "freq",
                                     "community", "coreness", "participation", "n_repos")}} for n in top_nodes],
            "aretes": ar_fortes[:500]},
        "APPENDIX_STYLOMETRIE": A["style"].to_dict("records") if len(A.get("style", [])) else [],
        "APPENDIX_REFERENCES_INTER_DEPOTS": cor.references.to_dict("records"),
        "APPENDIX_SIMILARITES": (A["similarite"]["paires"] if A.get("similarite") else []),
        "APPENDIX_CODE": code_agg,
        "APPENDIX_DOUBLONS": {"fichiers_identiques": int(len(cor.doublons)),
                              "paires": (cor.doublons.groupby(["repo_orig", "repo_dup"]).size()
                                         .reset_index(name="n").to_dict("records") if len(cor.doublons) else [])},
        "APPENDIX_TODOS": {"total": int(len(cor.todos)),
                           "par_depot": (cor.todos["repo"].value_counts().head(15).to_dict() if len(cor.todos) else {}),
                           "exemples": (cor.todos.head(25).to_dict("records") if len(cor.todos) else [])},
    }
    out = CFG.output_dir / "GRIMOIRE_DU_GLITCH_RAW.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(grimoire, f, indent=2, ensure_ascii=False, default=_json_default)
    log(f"Grimoire forgé : {C.BWHITE}{out}", "ok")

    safe("tables CSV", exporter_tables, A, ag, G, cor, cdf)
    log("Tables CSV → grimoire_output/tables/", "ok")
    try:
        nx.write_graphml(G, CFG.output_dir / "graphe_ontologique.graphml")
        log("Export GraphML (Gephi / Cytoscape)", "ok")
    except Exception as e:
        log(f"GraphML échec : {e}", "warn")
    safe("résumé LLM", ecrire_resume_markdown, grimoire, A, ag)
    return grimoire


def ecrire_resume_markdown(g, A, ag):
    """Version condensée (≈ quelques milliers de tokens) à coller dans un LLM : l'essentiel du corpus."""
    L = []
    add = L.append
    add("# GRIMOIRE DU GLITCH — résumé pour ingestion LLM\n")
    add(f"_Généré le {g['METADONNEES']['timestamp'][:19]} • {len(g['METADONNEES']['depots'])} dépôts_\n")
    add("> " + g["METADONNEES"]["principe_fondateur"] + "\n")
    m = g["APPENDIX_METRIQUES"]
    add(f"**Graphe** : {m.get('noeuds')} nœuds, {m.get('aretes')} arêtes, modularité "
        f"{g['LIBER_QUINTUS_COMMUNAUTES'].get('modularite')}, k-core max {m.get('k_core_max')}.\n")
    add("## Solutions idéales (nœuds centraux)")
    for e in g["LIBER_QUARTUS_SOLUTIONS_IDEALES"][:15]:
        add(f"- **{e['entite']}** — PR {e['pagerank']}, dans {e['depots']} dépôts, communauté {e['communaute']}, "
            f"participation {e['participation']}")
    add("\n## Communautés")
    for c, lab in list(g["LIBER_QUINTUS_COMMUNAUTES"]["etiquettes"].items())[:10]:
        add(f"- **C{c}** ({lab}) : " + ", ".join(pretty(x) for x in g["LIBER_QUINTUS_COMMUNAUTES"]["membres"][c][:12]))
    if g["LIBER_SEXTUS_THEMES_SEMANTIQUES"]:
        add("\n## Thèmes sémantiques")
        for t in sorted(g["LIBER_SEXTUS_THEMES_SEMANTIQUES"], key=lambda x: -x["taille"]):
            dom = ", ".join(d["depot"] for d in t["depots_dominants"][:3])
            add(f"- **{t['label']}** ({t['taille']} passages ; {dom}) — « {t['passage_prototype']['extrait'][:180]}… »")
    if g["LIBER_TERTIUS_TOPICS"]:
        add("\n## Topics LDA")
        for t in g["LIBER_TERTIUS_TOPICS"]:
            add(f"- T{t['topic_id']} : " + ", ".join(t["mots"][:8]))
    add("\n## Signature lexicale de chaque dépôt (log-odds)")
    for r, lst in g["LIBER_SEPTIMUS_TERMES_DISTINCTIFS"].items():
        add(f"- **{r}** : " + ", ".join(t["terme"] for t in lst[:8]))
    if g["LIBER_OCTAVUS_COLLOCATIONS"]:
        add("\n## Collocations")
        add(", ".join(c["bigramme"] for c in g["LIBER_OCTAVUS_COLLOCATIONS"][:25]))
    ner = g["LIBER_SECUNDUS_ENTITES"].get("par_type", {})
    if ner:
        add("\n## Entités nommées")
        for t, lst in ner.items():
            add(f"- {t} : " + ", ".join(f"{x['entite']} ({x['frequence']})" for x in lst[:8]))
    if g["APPENDIX_SIMILARITES"]:
        add("\n## Dépôts proches (forks / réutilisations)")
        for p in g["APPENDIX_SIMILARITES"][:10]:
            add(f"- {p['a']} ⇄ {p['b']} : cosinus {p['cosinus']}, fichiers identiques {p['fichiers_identiques_jaccard']}")
    if g["APPENDIX_REFERENCES_INTER_DEPOTS"]:
        add("\n## Références inter-dépôts (qui cite qui)")
        for r in sorted(g["APPENDIX_REFERENCES_INTER_DEPOTS"], key=lambda x: -x["mentions"])[:12]:
            add(f"- {r['source']} → {r['cible']} ({r['mentions']})")
    t = g["LIBER_NONUS_TEMPORALITE"]
    if t:
        add(f"\n## Temporalité\n{t['commits']} commits ; mots récurrents des messages : "
            + ", ".join(w for w, _ in t["mots_messages"][:15]))
    p = CFG.output_dir / "GRIMOIRE_RESUME.md"
    p.write_text("\n".join(L), encoding="utf-8")
    log(f"Résumé LLM : {p}", "ok")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — DATAVIZ STATIQUE (PNG, thème cosmique)
# ══════════════════════════════════════════════════════════════════════════════

FIGS: list[tuple[str, str, Path]] = []   # (titre, légende, chemin) → galerie du rapport HTML


def _save(fig, nom, titre, legende=""):
    out = CFG.output_dir / nom
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="#0a0a12")
    plt.close(fig)
    FIGS.append((titre, legende, out))
    log(f"Sauvegardé : {out}", "ok")


def _titre(ax, txt, size=13):
    ax.set_title(txt, color="#00fff7", fontsize=size, weight="bold", pad=12)


def _layout_communautes(G, partition, seed=42):
    """Layout par communautés : chaque communauté a sa zone, les nœuds s'organisent à l'intérieur."""
    comms = defaultdict(list)
    for n in G.nodes():
        comms[partition.get(n, -1)].append(n)
    meta = nx.Graph()
    meta.add_nodes_from(comms)
    for u, v, d in G.edges(data=True):
        a, b = partition.get(u, -1), partition.get(v, -1)
        if a != b:
            w = meta[a][b]["weight"] + d["weight"] if meta.has_edge(a, b) else d["weight"]
            meta.add_edge(a, b, weight=w)
    cpos = nx.spring_layout(meta, k=2.5 / max(len(comms), 1) ** 0.5, seed=seed, weight="weight") \
        if len(comms) > 1 else {c: np.zeros(2) for c in comms}
    pos = {}
    for c, nodes in comms.items():
        sub = G.subgraph(nodes)
        rayon = 0.35 + 0.07 * math.sqrt(len(nodes))
        loc = nx.spring_layout(sub, seed=seed, weight="weight", k=0.9 / max(len(nodes), 1) ** 0.5) \
            if len(nodes) > 1 else {nodes[0]: np.zeros(2)}
        for n, p in loc.items():
            pos[n] = cpos[c] * 2.2 + np.asarray(p) * rayon * 2
    return pos


def _labels(ax, pos, noms, size=9, couleur="#f0f0ff"):
    textes = []
    for n, txt in noms.items():
        x, y = pos[n]
        textes.append(ax.text(x, y, txt, fontsize=size, color=couleur, weight="bold", ha="center", va="center",
                              bbox=dict(boxstyle="round,pad=0.15", fc="#0a0a12", ec="none", alpha=0.65)))
    if HAS_ADJUST and textes:
        try:
            adjust_text_mod.adjust_text(textes, ax=ax)
        except Exception:
            pass


def viz_mousse_quantique(G, ag):
    section("Visualisation : Mousse Quantique (vue globale)", C.BCYAN, "🎨")
    B = noyau_visuel(G, ag)
    if B.number_of_nodes() < 5:
        log("Graphe trop petit", "warn")
        return
    part, pr = ag["partition"], ag["pagerank"]
    pos = _layout_communautes(B, part, CFG.seed)
    fig, ax = plt.subplots(figsize=(20, 20))
    edges = list(B.edges(data=True))
    w = np.array([d["weight"] for _, _, d in edges], float)
    widths = 0.3 + 3 * (w - w.min()) / (np.ptp(w) or 1)
    nx.draw_networkx_edges(B, pos, ax=ax, width=widths, alpha=0.22, edge_color="#5060b0")
    nodes = list(B.nodes())
    pmax = max(pr.get(n, 0) for n in nodes) or 1
    sizes = [70 + 1300 * (max(pr.get(n, 0), 0) / pmax) ** 0.7 for n in nodes]
    cols = [couleur_idx(part.get(n, -1)) for n in nodes]
    ec = ["#ffffff" if B.nodes[n].get("trame") else "#202040" for n in nodes]  # liseré blanc = lexique Trame
    nx.draw_networkx_nodes(B, pos, nodelist=nodes, node_size=sizes, node_color=cols, ax=ax,
                           edgecolors=ec, linewidths=[1.6 if e == "#ffffff" else 0.4 for e in ec], alpha=0.92)
    top = [n for n, _ in sorted(pr.items(), key=lambda x: -x[1])[:45] if n in B]
    _labels(ax, pos, {n: pretty(n) for n in top}, 9)
    for c in ag["communautes_top"][:10]:  # étiquette de communauté au barycentre
        pts = np.array([pos[n] for n in B if part.get(n) == c])
        if len(pts) >= 4:
            ax.text(*pts.mean(0), f"C{c}", fontsize=26, color=couleur_idx(c), alpha=0.18,
                    ha="center", va="center", weight="bold")
    ax.set_title("◆ NOYAU NON-LINÉAIRE — CARTOGRAPHIE DES SOLUTIONS IDÉALES ◆", color="#00fff7",
                 fontsize=22, weight="bold", pad=25)
    mod = ag.get("modularite")
    fig.text(0.5, 0.075, f"Nœuds : {B.number_of_nodes()}  │  Arêtes (backbone) : {B.number_of_edges()}  │  "
                         f"Modularité : {mod:.4f}  │  Communautés : {len(set(part.values()))}  │  liseré blanc = lexique Trame"
             if mod is not None else "", ha="center", color="#8080c0", fontsize=11, style="italic")
    ax.axis("off")
    _save(fig, "mousse_quantique_noyau.png", "Mousse quantique",
          "Backbone (arbre couvrant max ∪ top-3 voisins). Taille = PageRank, couleur = communauté.")


def viz_ego_networks(G, ag, top_k=3, max_voisins=35):
    section("Visualisation : Ego-networks des Dieux", C.BMAGENTA, "🎨")
    pr, part = ag["pagerank"], ag["partition"]
    top = [n for n, _ in sorted(pr.items(), key=lambda x: -x[1])[:top_k]]
    if not top:
        return
    fig, axes = plt.subplots(1, len(top), figsize=(9 * len(top), 9), squeeze=False)
    for ax, dieu in zip(axes[0], top):
        # radius=2 sur un hub = milliers de nœuds illisibles : on garde les voisins les plus liés
        v1 = sorted(G[dieu].items(), key=lambda kv: -kv[1]["weight"])[:max_voisins]
        noyau = {dieu} | {v for v, _ in v1}
        v2 = Counter()
        for v, _ in v1[:12]:
            for x, d in G[v].items():
                if x not in noyau:
                    v2[x] += d["weight"]
        ego = G.subgraph(noyau | {x for x, _ in v2.most_common(25)})
        pos = nx.spring_layout(ego, k=1.0 / math.sqrt(len(ego)), seed=CFG.seed, weight="weight")
        sizes = [100 + pr.get(n, 0) * 20000 for n in ego.nodes()]
        cols = ["#ff60a0" if n == dieu else couleur_idx(part.get(n, -1)) for n in ego.nodes()]
        nx.draw_networkx_edges(ego, pos, ax=ax, alpha=0.25, edge_color="#6070b0",
                               width=[0.4 + math.log1p(d["weight"]) * 0.4 for _, _, d in ego.edges(data=True)])
        nx.draw_networkx_nodes(ego, pos, ax=ax, node_size=sizes, node_color=cols, edgecolors="#ffffff", linewidths=0.4)
        _labels(ax, pos, {n: pretty(n) for n in ego.nodes() if n in noyau}, 8, "#e0e0ff")
        ax.set_title(f"◉ {pretty(dieu)}", color="#ff60a0", fontsize=14, weight="bold")
        ax.axis("off")
    fig.suptitle("EGO-NETWORKS DES SOLUTIONS IDÉALES", color="#00fff7", fontsize=18, weight="bold", y=1.0)
    _save(fig, "ego_networks.png", "Ego-networks", "Voisins directs les plus liés + quelques voisins de rang 2.")


def viz_heatmap_centralites(G, ag, top_n=25):
    section("Visualisation : Heatmap des centralités", C.BYELLOW, "🎨")
    pr = ag["pagerank"]
    top = [n for n, _ in sorted(pr.items(), key=lambda x: -x[1])[:top_n]]
    if len(top) < 3:
        return
    cols = [("PageRank", pr), ("Betweenness", ag["betweenness"]), ("Eigenvector", ag["eigenvector"]),
            ("Degré pond.", ag["degree"]), ("K-core", ag["coreness"]), ("Participation", ag["participation"])]
    data = np.array([[c[1].get(n, 0) for c in cols] for n in top], float)
    # rang-percentile : les centralités sont à queue lourde, /max écrasait tout sauf le n°1
    allv = [np.array(list(c[1].values()), float) for c in cols]
    norm = np.array([[ (allv[j] <= data[i, j]).mean() for j in range(len(cols))] for i in range(len(top))])
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.38)))
    im = ax.imshow(norm, cmap=NEON_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], color="#e0e0ff", fontsize=10, rotation=20)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([pretty(n) for n in top], color="#e0e0ff", fontsize=10)
    for i in range(len(top)):
        for j in range(len(cols)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#0a0a12" if norm[i, j] > 0.5 else "#e0e0ff")
    plt.colorbar(im, ax=ax, label="Rang percentile dans tout le graphe")
    _titre(ax, "◆ CENTRALITÉS — TOP NŒUDS (rang percentile) ◆", 14)
    _save(fig, "heatmap_centralites.png", "Heatmap des centralités", "Percentile 1.0 = parmi les plus centraux du graphe.")


def viz_wordclouds(A):
    if not HAS_WORDCLOUD:
        log("wordcloud non installé — skip (pip install wordcloud)", "warn")
        return
    section("Visualisation : Nuages de mots", C.BGREEN, "🎨")
    WC = wordcloud_mod.WordCloud
    freq = defaultdict(float)
    for k in A.get("keybert", []):
        freq[k["concept"]] += k.get("score", 1) * 2
    for k in A.get("tfidf", []):
        freq[k["concept"]] += k.get("score", 1) / max(A["tfidf"][0]["score"], 1e-9)
    if freq:
        wc = WC(width=1800, height=1000, background_color="#0a0a12", colormap="cool", max_words=120,
                prefer_horizontal=0.85, contour_width=2, contour_color="#00fff7").generate_from_frequencies(dict(freq))
        fig = plt.figure(figsize=(18, 10))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("◆ VRAIS NOMS — NUAGE ONTOLOGIQUE ◆", color="#00fff7", fontsize=20, weight="bold", pad=20)
        _save(fig, "wordcloud_vrais_noms.png", "Nuage ontologique", "Mots-clés sémantiques + TF-IDF.")
    dist = A.get("distinctifs", {})
    repos = list(dist)[:12]
    if len(repos) >= 4:  # mosaïque : signature lexicale de chaque dépôt (z-score log-odds)
        cols = 4
        rows = math.ceil(len(repos) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, r in zip(axes.ravel(), repos):
            fr = {t["terme"].replace(" ", "_"): max(t["z"], 0.1) for t in dist[r]}
            try:
                wc = WC(width=700, height=420, background_color="#0a0a12", colormap="spring",
                        max_words=15, prefer_horizontal=0.9).generate_from_frequencies(fr)
                ax.imshow(wc, interpolation="bilinear")
            except Exception:
                pass
            ax.set_title(r[:30], color="#ff60a0", fontsize=11, weight="bold")
        fig.suptitle("SIGNATURE LEXICALE PAR DÉPÔT (log-odds)", color="#00fff7", fontsize=18, weight="bold", y=1.0)
        _save(fig, "wordclouds_par_depot.png", "Signature lexicale par dépôt",
              "Mots sur-représentés dans un dépôt par rapport au reste du corpus (z-score).")


def viz_distinctifs_barres(A):
    d = A.get("distinctifs", {})
    repos = list(d)[:12]
    if len(repos) < 2:
        return
    section("Visualisation : termes distinctifs (barres)", C.BGREEN, "🎨")
    cols = 3
    rows = math.ceil(len(repos) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (ax, r) in enumerate(zip(axes.ravel(), repos)):
        ax.axis("on")
        t = d[r][:8][::-1]
        ax.barh([x["terme"] for x in t], [x["z"] for x in t], color=couleur_idx(i))
        ax.set_title(r[:34], color=couleur_idx(i), fontsize=11, weight="bold")
        ax.tick_params(labelsize=8)
    fig.suptitle("TERMES DISTINCTIFS PAR DÉPÔT (z-score)", color="#00fff7", fontsize=16, weight="bold")
    fig.tight_layout()
    _save(fig, "termes_distinctifs.png", "Termes distinctifs", "z-score du log-odds : plus c'est haut, plus le terme est spécifique.")


def viz_topics(A):
    topics, info = A.get("topics"), A.get("lda_info")
    if not topics:
        return
    section("Visualisation : Topics LDA", C.BMAGENTA, "🎨")
    k = len(topics)
    cols = min(4, k)
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for t, ax in zip(topics, axes.ravel()):
        ax.axis("on")
        mots, poids = t["mots"][:8][::-1], t["poids"][:8][::-1]
        ax.barh(mots, poids, color=couleur_idx(t["topic_id"]))
        ax.set_title(f"T{t['topic_id']} • {t['prevalence'] * 100:.0f}%", color=couleur_idx(t["topic_id"]), fontsize=11)
        ax.tick_params(labelsize=8)
    fig.suptitle(f"TOPICS LATENTS (LDA, k={k} choisi par cohérence NPMI)", color="#00fff7", fontsize=16, weight="bold")
    fig.tight_layout()
    _save(fig, "topics_lda.png", "Topics LDA", "k choisi automatiquement par cohérence NPMI.")
    if info is not None and len(info["repos"]) >= 2:
        mat = info["matrice"]
        mat = mat / mat.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(1 + 0.9 * k, 1 + 0.38 * len(info["repos"])))
        im = ax.imshow(mat, cmap=COSMIC_CMAP, aspect="auto")
        ax.set_xticks(range(k))
        ax.set_xticklabels([f"T{t['topic_id']}" for t in topics], color="#e0e0ff")
        ax.set_yticks(range(len(info["repos"])))
        ax.set_yticklabels([r[:32] for r in info["repos"]], color="#e0e0ff", fontsize=9)
        plt.colorbar(im, ax=ax, label="part du dépôt")
        _titre(ax, "◆ TOPICS × DÉPÔTS ◆")
        _save(fig, "topics_x_depots.png", "Topics × dépôts", "Quelle part de chaque dépôt relève de quel topic.")


def viz_carte_semantique(A):
    sem = A.get("sem")
    if not sem:
        return
    section("Visualisation : Carte sémantique 2D", C.BCYAN, "🎨")
    xy, lab, passages = sem["xy"], sem["labels"], A["passages"]
    fig, ax = plt.subplots(figsize=(15, 11))
    for c in range(sem["k"]):
        m = lab == c
        ax.scatter(xy[m, 0], xy[m, 1], s=9, alpha=0.55, color=couleur_idx(c), linewidths=0)
    textes = []
    for t in sem["themes"]:
        c = t["theme_id"]
        pts = xy[lab == c]
        if len(pts):
            med = np.median(pts, axis=0)
            textes.append(ax.text(*med, f"{c}: {t['label'][:38]}", color="#ffffff", fontsize=10, weight="bold",
                                  ha="center", va="center",
                                  bbox=dict(boxstyle="round,pad=0.25", fc=couleur_idx(c), ec="none", alpha=0.55)))
    if HAS_ADJUST:
        try:
            adjust_text_mod.adjust_text(textes, ax=ax)
        except Exception:
            pass
    ax.axis("off")
    _titre(ax, f"◆ CARTE SÉMANTIQUE DU CORPUS — {sem['k']} THÈMES ({len(xy)} passages) ◆", 16)
    _save(fig, "carte_semantique.png", "Carte sémantique", f"Chaque point = un passage ; projection 2D des embeddings ({'UMAP' if HAS_UMAP else 't-SNE/SVD'}).")

    # même carte, colorée par dépôt (top 12)
    repos = [r for r, _ in Counter(p["repo"] for p in passages).most_common(12)]
    rep_pts = np.array([passages[i]["repo"] for i in A["rep_sel"]])
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.scatter(xy[:, 0], xy[:, 1], s=6, color="#303050", alpha=0.4, linewidths=0)
    for i, r in enumerate(repos):
        m = rep_pts == r
        ax.scatter(xy[m, 0], xy[m, 1], s=12, alpha=0.75, color=couleur_idx(i), label=r[:30], linewidths=0)
    ax.legend(fontsize=8, facecolor="#10101c", edgecolor="#2a2a3a", labelcolor="#d0d0e0", markerscale=2, loc="best")
    ax.axis("off")
    _titre(ax, "◆ CARTE SÉMANTIQUE — COLORÉE PAR DÉPÔT ◆", 16)
    _save(fig, "carte_semantique_depots.png", "Carte sémantique par dépôt", "Recouvrement thématique entre dépôts.")


def viz_themes_x_depots(A):
    sem = A.get("sem")
    if not sem or len(sem["repos"]) < 2:
        return
    section("Visualisation : Thèmes × dépôts", C.BYELLOW, "🎨")
    comp = sem["composition"]
    keep = np.argsort(-comp.sum(1))[:20]
    mat = comp[keep] / np.maximum(comp[keep].sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.5 + 1.0 * sem["k"], 1 + 0.4 * len(keep)))
    im = ax.imshow(mat, cmap=COSMIC_CMAP, aspect="auto")
    ax.set_xticks(range(sem["k"]))
    ax.set_xticklabels([f"{t['theme_id']}: {t['label'][:18]}" for t in sem["themes"]], rotation=45, ha="right",
                       color="#e0e0ff", fontsize=8)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([sem["repos"][i][:32] for i in keep], color="#e0e0ff", fontsize=9)
    plt.colorbar(im, ax=ax, label="part du dépôt")
    _titre(ax, "◆ THÈMES SÉMANTIQUES × DÉPÔTS ◆")
    _save(fig, "themes_x_depots.png", "Thèmes × dépôts", "Composition thématique de chaque dépôt.")


def viz_similarite_depots(A):
    s = A.get("similarite")
    if not s:
        return
    section("Visualisation : Similarité entre dépôts", C.BMAGENTA, "🎨")
    o = s["ordre"]
    labs = [s["repos"][i][:30] for i in o]
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    for ax, mat, nom, cmap in ((axes[0], s["sim"], "Similarité lexicale (TF-IDF cosinus)", COSMIC_CMAP),
                               (axes[1], s["jaccard_hash"], "Fichiers identiques (Jaccard)", NEON_CMAP)):
        m = mat[np.ix_(o, o)]
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(o)))
        ax.set_xticklabels(labs, rotation=90, fontsize=8, color="#e0e0ff")
        ax.set_yticks(range(len(o)))
        ax.set_yticklabels(labs, fontsize=8, color="#e0e0ff")
        plt.colorbar(im, ax=ax, fraction=0.04)
        _titre(ax, nom, 13)
    fig.suptitle("◆ PARENTÉ ENTRE DÉPÔTS (ordre = clustering hiérarchique) ◆", color="#00fff7", fontsize=17, weight="bold")
    fig.tight_layout()
    _save(fig, "similarite_depots.png", "Parenté entre dépôts", "Les blocs révèlent forks, copies et familles thématiques.")


def viz_stylometrie(A):
    st = A.get("style")
    if st is None or len(st) < 3:
        return
    section("Visualisation : Stylométrie", C.BGREEN, "🎨")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    ax = axes[0]
    sc = ax.scatter(st["mattr"], st["mots_par_phrase"], s=40 + st["tokens"] / st["tokens"].max() * 900,
                    c=st["densite_trame_pm"], cmap=NEON_CMAP, alpha=0.85, edgecolors="#ffffff", linewidths=0.5)
    for _, r in st.iterrows():
        ax.annotate(r["repo"][:22], (r["mattr"], r["mots_par_phrase"]), fontsize=8, color="#e0e0ff",
                    xytext=(5, 4), textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="densité lexique Trame (‰ lemmes)")
    ax.set_xlabel("MATTR — diversité lexicale")
    ax.set_ylabel("mots par phrase")
    ax.grid(alpha=0.3)
    _titre(ax, "Carte stylométrique (taille = volume)", 13)
    ax = axes[1]
    s2 = st.sort_values("densite_trame_pm")
    ax.barh(s2["repo"].str[:28], s2["densite_trame_pm"], color=[couleur_idx(i) for i in range(len(s2))])
    ax.set_xlabel("‰ de lemmes appartenant au LEXIQUE_TRAME")
    ax.tick_params(labelsize=8)
    _titre(ax, "Densité du lexique Trame par dépôt", 13)
    fig.tight_layout()
    _save(fig, "stylometrie.png", "Stylométrie", "Diversité lexicale, longueur de phrase et saturation en vocabulaire Trame.")


def viz_zipf(A):
    L = A["L"]
    f = np.array(sorted(L.tok_count.values(), reverse=True), float)
    if len(f) < 100:
        return
    section("Visualisation : Loi de Zipf", C.BCYAN, "🎨")
    r = np.arange(1, len(f) + 1)
    pente = pente_zipf(list(f))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(r, f, ".", ms=3, color="#00fff7", alpha=0.7, label="corpus")
    if pente is not None:
        hi = min(len(f), 1500)
        a = np.polyfit(np.log(r[2:hi]), np.log(f[2:hi]), 1)
        ax.loglog(r[2:hi], np.exp(a[1]) * r[2:hi] ** a[0], color="#ff60a0", lw=2, label=f"pente = {a[0]:.2f}")
    top = np.argsort(-f)[:12]
    ivoc = [w for w, _ in L.tok_count.most_common(12)]
    for i, (w, _) in enumerate(L.tok_count.most_common(12)):
        ax.annotate(pretty(L.ivocab[w]), (r[i], f[i]), fontsize=8, color="#ffd060", xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("rang")
    ax.set_ylabel("fréquence")
    ax.grid(alpha=0.2, which="both")
    ax.legend(facecolor="#10101c", edgecolor="#2a2a3a", labelcolor="#d0d0e0")
    _titre(ax, "◆ LOI DE ZIPF DU CORPUS (lemmes de contenu) ◆")
    _save(fig, "zipf.png", "Loi de Zipf", "Une pente proche de −1 = langage naturel ; très différente = vocabulaire artificiel/répétitif.")


def viz_ner(A):
    ner = A.get("ner", {})
    if not ner.get("top_entites"):
        return
    section("Visualisation : Entités nommées", C.BMAGENTA, "🎨")
    types = list(ner["par_type"])[:4]
    fig, axes = plt.subplots(1, len(types), figsize=(6 * len(types), 7), squeeze=False)
    for i, (ax, t) in enumerate(zip(axes[0], types)):
        lst = ner["par_type"][t][:12][::-1]
        ax.barh([x["entite"][:26] for x in lst], [x["frequence"] for x in lst], color=couleur_idx(i))
        ax.set_title(t, color=couleur_idx(i), fontsize=13, weight="bold")
        ax.tick_params(labelsize=9)
    fig.suptitle("ENTITÉS NOMMÉES PAR TYPE", color="#00fff7", fontsize=16, weight="bold")
    fig.tight_layout()
    _save(fig, "entites.png", "Entités nommées", "Personnes, lieux, organisations, divers.")


def viz_inventaire(cor):
    inv = cor.inventaire
    if inv is None or len(inv) == 0:
        return
    section("Visualisation : Inventaire des dépôts", C.BCYAN, "🎨")
    piv = inv.pivot_table(index="repo", columns="ext", values="octets", aggfunc="sum", fill_value=0)
    top_ext = piv.sum().sort_values(ascending=False).index[:9]
    piv["autres"] = piv.drop(columns=top_ext).sum(axis=1)
    piv = piv[list(top_ext) + ["autres"]]
    piv = piv.loc[piv.sum(axis=1).sort_values().index[-25:]]
    fig, ax = plt.subplots(figsize=(13, 0.4 * len(piv) + 3))
    gauche = np.zeros(len(piv))
    for i, e in enumerate(piv.columns):
        ax.barh(piv.index.str[:34], piv[e] / 1024, left=gauche, color=couleur_idx(i), label=e)
        gauche += piv[e].values / 1024
    ax.set_xscale("log")
    ax.set_xlabel("Ko (échelle log)")
    ax.legend(ncol=5, fontsize=8, facecolor="#10101c", edgecolor="#2a2a3a", labelcolor="#d0d0e0", loc="lower right")
    _titre(ax, "◆ INVENTAIRE : OCTETS PAR DÉPÔT ET EXTENSION ◆")
    _save(fig, "inventaire.png", "Inventaire", "Tous fichiers confondus (pas seulement le texte analysé).")


def viz_temporalite(A):
    cm = A.get("commits")
    if not cm:
        return
    section("Visualisation : Temporalité (commits)", C.BYELLOW, "🎨")
    men = cm["mensuel"]
    if len(men) >= 2:
        top = men.sum().sort_values(ascending=False).index[:10]
        autres = men.drop(columns=top).sum(axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(21, 7), gridspec_kw={"width_ratios": [2.2, 1]})
        ax = axes[0]
        bas = np.zeros(len(men))
        for i, r in enumerate(top):
            ax.bar(men.index, men[r], bottom=bas, width=22, color=couleur_idx(i), label=r[:26])
            bas += men[r].values
        if autres.sum() > 0:
            ax.bar(men.index, autres, bottom=bas, width=22, color="#404060", label="autres")
        ax.legend(fontsize=8, ncol=2, facecolor="#10101c", edgecolor="#2a2a3a", labelcolor="#d0d0e0")
        ax.set_ylabel("commits / mois")
        _titre(ax, "Activité mensuelle par dépôt", 13)
        ax = axes[1]
        im = ax.imshow(cm["rythme"], cmap=COSMIC_CMAP, aspect="auto")
        ax.set_yticks(range(7))
        ax.set_yticklabels(["lun", "mar", "mer", "jeu", "ven", "sam", "dim"])
        ax.set_xlabel("heure (UTC)")
        plt.colorbar(im, ax=ax)
        _titre(ax, "Rythme hebdomadaire", 13)
        fig.tight_layout()
        _save(fig, "temporalite.png", "Temporalité", "Quand le multivers s'est écrit : mois, jours, heures.")
    pr = cm["par_repo"]
    if len(pr) >= 3:
        pr = pr.sort_values("debut")
        fig, ax = plt.subplots(figsize=(14, 0.38 * len(pr) + 2))
        for i, (_, r) in enumerate(pr.iterrows()):
            ax.plot([r["debut"], r["fin"]], [i, i], color=couleur_idx(i), lw=6, solid_capstyle="round", alpha=0.85)
            ax.text(r["fin"], i, f"  {r['commits']}", va="center", color="#e0e0ff", fontsize=8)
        ax.set_yticks(range(len(pr)))
        ax.set_yticklabels([x[:32] for x in pr["repo"]], fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        _titre(ax, "◆ DURÉE DE VIE DES DÉPÔTS (1er → dernier commit, n = commits) ◆")
        _save(fig, "durees_de_vie.png", "Durée de vie des dépôts", "Frise chronologique du multivers.")


def viz_references(cor):
    ref = cor.references
    if ref is None or len(ref) < 2:
        return
    section("Visualisation : Références inter-dépôts", C.BGREEN, "🎨")
    D = nx.DiGraph()
    for _, r in ref.iterrows():
        D.add_edge(r["source"], r["cible"], weight=int(r["mentions"]))
    pos = nx.spring_layout(D, k=1.4 / math.sqrt(len(D)), seed=CFG.seed, weight="weight")
    ind = dict(D.in_degree(weight="weight"))
    fig, ax = plt.subplots(figsize=(15, 12))
    nx.draw_networkx_edges(D, pos, ax=ax, alpha=0.45, edge_color="#6a7ad0", arrows=True, arrowsize=14,
                           width=[0.6 + math.log1p(d["weight"]) for _, _, d in D.edges(data=True)],
                           connectionstyle="arc3,rad=0.12", node_size=1200)
    nx.draw_networkx_nodes(D, pos, ax=ax, node_size=[300 + ind.get(n, 0) * 40 for n in D],
                           node_color=[couleur_idx(i) for i, _ in enumerate(D)], edgecolors="#ffffff", linewidths=0.6)
    _labels(ax, pos, {n: n[:28] for n in D}, 9)
    ax.axis("off")
    _titre(ax, "◆ QUI CITE QUI — RÉSEAU DE RÉFÉRENCES INTER-DÉPÔTS ◆", 16)
    _save(fig, "references_inter_depots.png", "Références inter-dépôts", "Flèche A → B : A mentionne B (nom ou URL). Taille = nombre de citations reçues.")


def viz_communautes_matrice(ag):
    mat, top = ag.get("matrice_comm"), ag.get("communautes_top")
    if mat is None or len(top) < 3:
        return
    section("Visualisation : Flux entre communautés", C.BBLUE, "🎨")
    lab = [f"C{c}: {ag['etiquettes_comm'][c][:26]}" for c in top]
    m = np.log1p(mat)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(m, cmap=NEON_CMAP)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=8, color="#e0e0ff")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(lab, fontsize=8, color="#e0e0ff")
    plt.colorbar(im, ax=ax, label="log(1 + poids des arêtes)")
    _titre(ax, "◆ LIENS ENTRE COMMUNAUTÉS (diagonale = cohésion interne) ◆", 13)
    _save(fig, "communautes_matrice.png", "Liens entre communautés", "Quelles familles de concepts se parlent.")


def viz_code(cor):
    cd = cor.code
    if cd is None or len(cd) < 3:
        return
    section("Visualisation : Code Python", C.BYELLOW, "🎨")
    agg = cd.groupby("repo").agg(loc=("loc", "sum"), doc=("doc_cov", "mean"), cx=("complexite", "sum"),
                                 fn=("fonctions", "sum")).reset_index()
    agg = agg[agg["loc"] > 0]
    if len(agg) < 3:
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(agg["loc"], agg["doc"].fillna(0) * 100, s=60 + agg["fn"] * 4,
                    c=agg["cx"] / agg["loc"], cmap=COSMIC_CMAP, edgecolors="#fff", linewidths=0.5, alpha=0.9)
    ax.set_xscale("log")
    for _, r in agg.iterrows():
        ax.annotate(r["repo"][:22], (r["loc"], (r["doc"] or 0) * 100), fontsize=8, color="#e0e0ff",
                    xytext=(5, 4), textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="complexité / ligne")
    ax.set_xlabel("lignes de code Python (log)")
    ax.set_ylabel("couverture docstrings (%)")
    ax.grid(alpha=0.3)
    _titre(ax, "◆ CODE : TAILLE × DOCUMENTATION × COMPLEXITÉ ◆", 14)
    _save(fig, "code_python.png", "Code Python", "Taille (log), couverture de docstrings et complexité des dépôts Python.")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 6 — DASHBOARD INTERACTIF (Plotly) + GRAPHE PYVIS + RAPPORT HTML
# ══════════════════════════════════════════════════════════════════════════════

def _pills(items):
    return "".join("<span class='pill'>" + html_lib.escape(str(x)) + "</span>" for x in items)


def _fig_html(fig, premier: bool):
    inc = True if (premier and CFG.offline_plotly) else ("cdn" if premier else False)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0a12", plot_bgcolor="#0a0a12",
                      font=dict(family="monospace", color="#d0d0e0"), margin=dict(l=40, r=20, t=60, b=40))
    return fig.to_html(full_html=False, include_plotlyjs=inc, config={"displaylogo": False, "responsive": True})


def figures_plotly(A, ag, G, cor, cdf) -> list[tuple[str, str, object]]:
    go = plotly_go
    figs = []
    # 1. Treemap dépôts → extensions
    inv = cor.inventaire
    if inv is not None and len(inv):
        g = inv.groupby(["repo", "ext"], as_index=False)["octets"].sum()
        ids, labels, parents, vals = ["Multivers"], ["Multivers"], [""], [0]
        for r, sub in g.groupby("repo"):
            ids.append(r); labels.append(r); parents.append("Multivers"); vals.append(0)
            for _, row in sub.iterrows():
                ids.append(f"{r}|{row['ext']}"); labels.append(row["ext"]); parents.append(r); vals.append(int(row["octets"]))
        fig = go.Figure(go.Treemap(ids=ids, labels=labels, parents=parents, values=vals, branchvalues="total",
                                   marker=dict(colorscale="Purples")))
        fig.update_layout(title="Treemap : octets par dépôt et extension")
        figs.append(("Inventaire", "Cliquer pour zoomer sur un dépôt.", fig))

    # 2. Carte sémantique interactive
    sem = A.get("sem")
    if sem:
        xy, lab, passages = sem["xy"], sem["labels"], A["passages"]
        fig = go.Figure()
        for t in sem["themes"]:
            c = t["theme_id"]
            idx = np.where(lab == c)[0]
            fig.add_trace(go.Scattergl(
                x=xy[idx, 0], y=xy[idx, 1], mode="markers", name=f"{c}: {t['label'][:34]}",
                marker=dict(size=6, color=couleur_idx(c), opacity=0.7),
                text=[f"<b>{html_lib.escape(passages[A['rep_sel'][i]]['repo'])}</b><br>"
                      f"{html_lib.escape(passages[A['rep_sel'][i]]['path'])}<br>"
                      f"{html_lib.escape(re.sub(chr(10), ' ', passages[A['rep_sel'][i]]['texte'][:200]))}…" for i in idx],
                hovertemplate="%{text}<extra></extra>"))
        fig.update_layout(title="Carte sémantique du corpus (survol = extrait ; légende cliquable)", height=720,
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        figs.append(("Carte sémantique", "Chaque point est un passage de texte.", fig))

        # Sankey dépôts → thèmes
        comp = sem["composition"]
        keep = np.argsort(-comp.sum(1))[:14]
        labels = [sem["repos"][i][:30] for i in keep] + [f"T{t['theme_id']}: {t['label'][:26]}" for t in sem["themes"]]
        src, tgt, val = [], [], []
        for a, i in enumerate(keep):
            for c in range(sem["k"]):
                if comp[i, c] > 0:
                    src.append(a); tgt.append(len(keep) + c); val.append(float(comp[i, c]))
        fig = go.Figure(go.Sankey(node=dict(label=labels, pad=12, thickness=14,
                                            color=["#7a5cff"] * len(keep) + [couleur_idx(c) for c in range(sem["k"])]),
                                  link=dict(source=src, target=tgt, value=val, color="rgba(120,130,220,0.25)")))
        fig.update_layout(title="Flux dépôts → thèmes sémantiques", height=640)
        figs.append(("Flux dépôts → thèmes", "Épaisseur = nombre de passages.", fig))

    # 3. Nœuds : PageRank × betweenness
    if ag.get("pagerank"):
        top = sorted(ag["pagerank"], key=lambda n: -ag["pagerank"][n])[:500]
        fig = go.Figure(go.Scattergl(
            x=[ag["pagerank"][n] for n in top], y=[ag["betweenness"].get(n, 0) for n in top], mode="markers",
            marker=dict(size=[6 + min(G.nodes[n]["freq"], 400) ** 0.5 * 1.6 for n in top],
                        color=[ag["partition"].get(n, 0) for n in top], colorscale="Turbo", opacity=0.8,
                        colorbar=dict(title="communauté")),
            text=[f"<b>{pretty(n)}</b><br>communauté {ag['partition'].get(n)}<br>k-core {ag['coreness'].get(n)}"
                  f"<br>participation {ag['participation'].get(n, 0):.2f}<br>{G.nodes[n]['n_repos']} dépôts" for n in top],
            hovertemplate="%{text}<extra></extra>"))
        fig.update_layout(title="Nœuds : PageRank (influence) × betweenness (pont) — taille = fréquence",
                          xaxis_title="PageRank", yaxis_title="betweenness", xaxis_type="log", height=680)
        figs.append(("Hubs & ponts", "En haut à droite : hubs-ponts ; en haut à gauche : ponts discrets.", fig))

    # 4. Similarité dépôts
    s = A.get("similarite")
    if s:
        o = s["ordre"]
        labs = [s["repos"][i] for i in o]
        fig = go.Figure(go.Heatmap(z=s["sim"][np.ix_(o, o)], x=labs, y=labs, colorscale="Magma", zmin=0, zmax=1))
        fig.update_layout(title="Similarité lexicale entre dépôts (clustering hiérarchique)", height=780)
        figs.append(("Parenté entre dépôts", "Blocs = familles de dépôts.", fig))

    # 5. Termes distinctifs avec sélecteur
    d = A.get("distinctifs", {})
    if d:
        fig = go.Figure()
        repos = list(d)
        for i, r in enumerate(repos):
            t = d[r][:12][::-1]
            fig.add_trace(go.Bar(x=[x["z"] for x in t], y=[x["terme"] for x in t], orientation="h",
                                 name=r, visible=(i == 0), marker_color=couleur_idx(i)))
        fig.update_layout(title="Termes distinctifs (z-score) — choisir un dépôt", height=520,
                          updatemenus=[dict(buttons=[dict(label=r[:34], method="update",
                                                          args=[{"visible": [j == i for j in range(len(repos))]}])
                                                     for i, r in enumerate(repos)], x=0, y=1.18, xanchor="left")])
        figs.append(("Signature de chaque dépôt", "Termes sur-représentés vs le reste du corpus.", fig))

    # 6. Temporalité
    cm = A.get("commits")
    if cm and len(cm["mensuel"]) >= 2:
        men = cm["mensuel"]
        fig = go.Figure()
        for i, r in enumerate(men.sum().sort_values(ascending=False).index[:15]):
            fig.add_trace(go.Bar(x=men.index, y=men[r], name=r[:28], marker_color=couleur_idx(i)))
        fig.update_layout(barmode="stack", title="Commits par mois et par dépôt", height=480)
        figs.append(("Temporalité", "", fig))

    # 7. Stylométrie
    st = A.get("style")
    if st is not None and len(st) >= 3:
        fig = go.Figure(go.Scatter(
            x=st["mattr"], y=st["mots_par_phrase"], mode="markers+text", text=st["repo"].str[:20],
            textposition="top center", textfont=dict(size=9),
            marker=dict(size=10 + 50 * st["tokens"] / st["tokens"].max(), color=st["densite_trame_pm"],
                        colorscale="Viridis", showscale=True, colorbar=dict(title="‰ Trame"), opacity=0.85),
            hovertemplate="%{text}<br>MATTR %{x}<br>%{y} mots/phrase<extra></extra>"))
        fig.update_layout(title="Carte stylométrique", xaxis_title="MATTR (diversité lexicale)",
                          yaxis_title="mots par phrase", height=560)
        figs.append(("Stylométrie", "", fig))
    return figs


def generer_html_interactif(G, ag, A):
    """Graphe pyvis : backbone lisible, forme = type d'entité, couleur = communauté."""
    if not HAS_PYVIS:
        log("pyvis non installé — skip graphe interactif (pip install pyvis)", "warn")
        return
    section("Génération : graphe interactif (pyvis)", C.BMAGENTA, "🌐")
    Network = pyvis_network.Network
    try:
        net = Network(height="900px", width="100%", bgcolor="#0a0a12", font_color="#e0e0ff",
                      notebook=False, directed=False, cdn_resources="in_line")
    except TypeError:
        net = Network(height="900px", width="100%", bgcolor="#0a0a12", font_color="#e0e0ff",
                      notebook=False, directed=False)
    net.barnes_hut(gravity=-9000, central_gravity=0.25, spring_length=130)
    B = noyau_visuel(G, ag)
    formes = {"PER": "star", "PERSON": "star", "LOC": "triangle", "GPE": "triangle", "ORG": "square"}
    for n in B.nodes():
        d = G.nodes[n]
        net.add_node(n, label=pretty(n), size=8 + ag["pagerank"].get(n, 0) * 400,
                     color=couleur_idx(ag["partition"].get(n, -1)), shape=formes.get(d.get("ner_type", ""), "dot"),
                     borderWidth=3 if d.get("trame") else 1,
                     title=(f"{pretty(n)}\nPageRank {ag['pagerank'].get(n, 0):.5f}\nbetweenness "
                            f"{ag['betweenness'].get(n, 0):.4f}\nk-core {ag['coreness'].get(n, 0)}\n"
                            f"communauté {ag['partition'].get(n)} — {ag['etiquettes_comm'].get(ag['partition'].get(n), '')}\n"
                            f"{d['n_repos']} dépôts • fréquence {d['freq']}"))
    for u, v, d in B.edges(data=True):
        net.add_edge(u, v, value=d.get("weight", 1), title=f"co-occurrences {d.get('weight')} • NPMI {d.get('npmi')}")
    try:
        net.show_buttons(filter_=["physics"])
    except Exception:
        pass
    chemin = CFG.output_dir / "graphe_interactif.html"
    try:
        net.write_html(str(chemin), notebook=False)
        log(f"Graphe interactif : {chemin}", "ok")
    except Exception as e:
        log(f"pyvis écriture échec : {e}", "warn")


def generer_rapport_html(A, ag, G, cor, cdf, grimoire, duree):
    section("Génération : rapport HTML consolidé", C.BGREEN, "🌐")
    esc = html_lib.escape
    figs = []
    if HAS_PLOTLY:
        figs = safe("figures plotly", figures_plotly, A, ag, G, cor, cdf, default=[])
    else:
        log("plotly non installé — rapport sans graphiques interactifs (pip install plotly)", "warn")

    m = ag.get("metriques", {})
    kpis = [("Dépôts", len({d["repo"] for d in cor.docs})), ("Documents", fmt_int(len(cor.docs))),
            ("Passages", fmt_int(len(A["passages"]))), ("Vocabulaire", fmt_int(len(A["L"].ivocab))),
            ("Nœuds", fmt_int(m.get("noeuds", 0))), ("Arêtes", fmt_int(m.get("aretes", 0))),
            ("Communautés", len(set(ag.get("partition", {}).values()))),
            ("Thèmes", (A.get("sem") or {}).get("k", "–")), ("Commits", (A.get("commits") or {}).get("n", "–"))]
    h = ["<!doctype html><html lang='fr'><head><meta charset='utf-8'><title>Grimoire du Glitch — rapport</title>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'><style>",
         ":root{--bg:#0a0a12;--fg:#d0d0e0;--c1:#00fff7;--c2:#ff60a0;--c3:#ffd060;--card:#12121f;--bd:#2a2a3a}",
         "*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-monospace,Menlo,Consolas,monospace}",
         "header{padding:2.2rem 2rem 1rem;border-bottom:1px solid var(--bd)}h1{margin:0;color:var(--c1);letter-spacing:.04em}",
         "header p{margin:.3rem 0 0;color:#8080a0;font-style:italic}nav{position:sticky;top:0;z-index:5;background:#0a0a12ee;padding:.6rem 2rem;border-bottom:1px solid var(--bd);display:flex;gap:1rem;flex-wrap:wrap}",
         "nav a{color:var(--c2);text-decoration:none;font-size:13px}main{padding:1rem 2rem 4rem;max-width:1500px;margin:auto}",
         ".kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.7rem;margin:1.2rem 0}",
         ".kpi{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:.8rem}.kpi b{display:block;font-size:22px;color:var(--c3)}",
         ".kpi span{font-size:12px;color:#8080a0}h2{color:var(--c2);border-bottom:1px solid var(--bd);padding-bottom:.3rem;margin-top:2.5rem}",
         "table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--bd);padding:.35rem .5rem;text-align:left;vertical-align:top}",
         "th{color:var(--c1)}.fig{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:.6rem;margin:1rem 0}",
         ".fig img{max-width:100%;height:auto;border-radius:8px}.cap{color:#8080a0;font-size:12px;margin:.2rem .4rem}",
         ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1rem}.pill{display:inline-block;background:#1c1c30;border:1px solid var(--bd);border-radius:99px;padding:0 .6rem;margin:.1rem;font-size:12px}",
         "</style></head><body>",
         "<header><h1>✦ GRIMOIRE DU GLITCH ✦</h1><p>« La réalité est une solution triviale d'un système non-linéaire. »"
         f" — généré le {datetime.now():%Y-%m-%d %H:%M} en {duree:.0f}s</p></header>",
         "<nav><a href='#kpi'>Chiffres</a><a href='#dieux'>Solutions idéales</a><a href='#themes'>Thèmes</a>"
         "<a href='#depots'>Dépôts</a><a href='#interactif'>Interactif</a><a href='#galerie'>Galerie</a>"
         "<a href='graphe_interactif.html'>Graphe pyvis ↗</a></nav><main>",
         "<section id='kpi'><div class='kpis'>" + "".join(f"<div class='kpi'><b>{v}</b><span>{k}</span></div>" for k, v in kpis)
         + "</div></section>"]

    h.append("<section id='dieux'><h2>Solutions idéales — nœuds les plus centraux</h2><table><tr><th>#</th><th>concept</th>"
             "<th>PageRank</th><th>betweenness</th><th>k-core</th><th>participation</th><th>dépôts</th><th>communauté</th></tr>")
    for i, e in enumerate(grimoire["LIBER_QUARTUS_SOLUTIONS_IDEALES"][:20], 1):
        lab = ag.get("etiquettes_comm", {}).get(e["communaute"], "")
        h.append(f"<tr><td>{i}</td><td><b>{esc(e['entite'])}</b></td><td>{e['pagerank']}</td><td>{e['betweenness']}</td>"
                 f"<td>{e['k_core']}</td><td>{e['participation']}</td><td>{e['depots']}</td><td>C{e['communaute']} <span class='cap'>{esc(lab)}</span></td></tr>")
    h.append("</table></section>")

    sem = A.get("sem")
    h.append("<section id='themes'><h2>Thèmes sémantiques</h2><div class='grid'>")
    for t in (sorted(sem["themes"], key=lambda x: -x["taille"]) if sem else []):
        pills = _pills((t["mots_cles_semantiques"] or t["mots_cles_ctfidf"])[:8])
        dom = ", ".join(esc(d["depot"]) for d in t["depots_dominants"][:3])
        h.append(f"<div class='kpi'><b style='font-size:15px;color:{couleur_idx(t['theme_id'])}'>{esc(t['label'])}</b>"
                 f"<span>{t['taille']} passages • {dom}</span><div>{pills}</div>"
                 f"<p class='cap'>« {esc(t['passage_prototype']['extrait'][:200])}… »</p></div>")
    if not sem:
        for t in A.get("topics", []):
            h.append(f"<div class='kpi'><b style='font-size:15px'>T{t['topic_id']} — {esc(t['label'])}</b>"
                     "<div>" + _pills(t["mots"][:8]) + "</div></div>")
    h.append("</div></section>")

    h.append("<section id='depots'><h2>Dépôts : signature lexicale & style</h2><table><tr><th>dépôt</th><th>termes distinctifs</th>"
             "<th>tokens</th><th>MATTR</th><th>‰ Trame</th></tr>")
    st = A.get("style")
    sty = {r["repo"]: r for _, r in st.iterrows()} if st is not None and len(st) else {}
    for r, lst in A.get("distinctifs", {}).items():
        s = sty.get(r)
        tok = fmt_int(s["tokens"]) if s is not None else ""
        mat = s["mattr"] if s is not None else ""
        den = s["densite_trame_pm"] if s is not None else ""
        h.append("<tr><td><b>" + esc(r) + "</b></td><td>" + _pills([t["terme"] for t in lst[:8]]) +
                 f"</td><td>{tok}</td><td>{mat}</td><td>{den}</td></tr>")
    h.append("</table></section>")

    h.append("<section id='interactif'><h2>Graphiques interactifs</h2>")
    for i, (titre, leg, fig) in enumerate(figs):
        h.append(f"<div class='fig'><div class='cap'><b>{esc(titre)}</b> {esc(leg)}</div>{_fig_html(fig, i == 0)}</div>")
    if not figs:
        h.append("<p class='cap'>(installez plotly pour activer cette section)</p>")
    h.append("</section><section id='galerie'><h2>Galerie</h2>")
    for titre, leg, chemin in FIGS:
        h.append(f"<div class='fig'><div class='cap'><b>{esc(titre)}</b> — {esc(leg)}</div>"
                 f"<a href='{esc(chemin.name)}'><img loading='lazy' src='{esc(chemin.name)}' alt='{esc(titre)}'></a></div>")
    h.append("</section></main></body></html>")
    out = CFG.output_dir / "rapport_grimoire.html"
    out.write_text("".join(h), encoding="utf-8")
    log(f"Rapport HTML : {out}", "ok")


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHÈSE & ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def afficher_synthese_finale(cor, A, ag, duree):
    banner("SYNTHÈSE FINALE DU RITUEL", "Le Glitch a parlé", couleur=C.BGREEN, symbole="✦")
    m = ag.get("metriques", {})
    section("Substance extraite", C.BCYAN, "▤")
    kv_line("Caractères", fmt_int(sum(s["extrait"] for s in cor.stats.values())))
    kv_line("Documents uniques", fmt_int(len(cor.docs)))
    kv_line("Passages analysés", fmt_int(len(A["passages"])))
    kv_line("Extensions", str(len([k for k, v in cor.stats.items() if v["fichiers"]])))
    section("Concepts extraits", C.BMAGENTA, "◈")
    kv_line("Mots-clés sémantiques", str(len(A.get("keybert", []))))
    kv_line("TF-IDF termes", str(len(A.get("tfidf", []))))
    kv_line("Entités NER", str(len(A.get("ner", {}).get("top_entites", []))))
    kv_line("Topics LDA", str(len(A.get("topics", []))))
    kv_line("Thèmes sémantiques", str((A.get("sem") or {}).get("k", 0)))
    kv_line("Collocations", str(len(A.get("collocations", []))))
    section("Graphe ontologique", C.BYELLOW, "◈")
    for k, v in m.items():
        kv_line(k.replace("_", " ").capitalize(), str(v))
    section("Artefacts générés", C.BGREEN, "📦")
    for f in sorted(CFG.output_dir.iterdir()):
        if f.is_dir():
            kv_line(f.name + "/", f"{len(list(f.iterdir()))} fichiers")
            continue
        t, unit = f.stat().st_size, "B"
        for u in ("KB", "MB"):
            if t > 1024:
                t /= 1024
                unit = u
        kv_line(f.name, f"{t:.1f} {unit}")
    section("Performance par étape", C.BCYAN, "⏱")
    for k, v in sorted(PHASE_TIMES.items(), key=lambda x: -x[1])[:8]:
        kv_line(k, f"{v:.1f}s")
    kv_line("Durée totale", f"{duree:.1f}s")
    kv_line("Terminé à", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"\n{C.BGREEN}{'═' * min(_term_width(), 100)}{C.RESET}")
    print(f"{C.BGREEN}{C.BOLD}  ✦ RITUEL TERMINÉ — ouvrez grimoire_output/rapport_grimoire.html ✦{C.RESET}")
    print(f"{C.BGREEN}{'═' * min(_term_width(), 100)}{C.RESET}\n")


def _ag_vide():
    return {"metriques": {}, "pagerank": {}, "betweenness": {}, "eigenvector": {}, "degree": {}, "partition": {},
            "modularite": None, "methode_comm": "n/a", "coreness": {}, "participation": {},
            "etiquettes_comm": {}, "communautes_top": [], "matrice_comm": None}


def ecrire_manifeste(repos, cor, duree):
    ver = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
           "networkx": nx.__version__, "spacy": getattr(spacy, "__version__", None) if HAS_SPACY else None,
           "sentence_transformers": getattr(sentence_transformers, "__version__", None) if HAS_ST else None}
    man = {"date": datetime.now(timezone.utc).isoformat(), "duree_s": round(duree, 1), "versions": ver,
           "parametres": {k: str(v) for k, v in vars(CFG).items() if k != "repos"},
           "depots": [{"label": r.label, "url": r.url, "head": r.head, "shallow": r.shallow} for r in repos],
           "etapes_s": {k: round(v, 2) for k, v in PHASE_TIMES.items()}, "documents": len(cor.docs)}
    (CFG.output_dir / "run_manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")


def parser_arguments(argv=None):
    p = argparse.ArgumentParser(description="Grimoire du Glitch v3 — text mining & dataviz sur dépôts GitHub")
    p.add_argument("--repos-file", help="fichier texte : une URL de dépôt par ligne (# = commentaire)")
    p.add_argument("--out", default=str(CFG.output_dir), help="dossier de sortie")
    p.add_argument("--work", default=str(CFG.work_dir), help="dossier des clones")
    p.add_argument("--force", action="store_true", help="re-clone tout, ignore le cache")
    p.add_argument("--ttl", type=float, default=CFG.cache_ttl_h, help="validité du cache de clone (heures)")
    p.add_argument("--no-history", action="store_true", help="clones shallow (pas de timeline)")
    p.add_argument("--workers", type=int, default=CFG.workers, help="clones en parallèle")
    p.add_argument("--no-embeddings", action="store_true", help="repli TF-IDF/SVD (sans sentence-transformers)")
    p.add_argument("--embed-model", default=CFG.embed_model)
    p.add_argument("--max-passages", type=int, default=CFG.max_passages)
    p.add_argument("--max-embed", type=int, default=CFG.max_embed)
    p.add_argument("--procs", type=int, default=1, help="processus spaCy (>1 : plus rapide, plus de RAM)")
    p.add_argument("--topics", type=int, default=0, help="nombre de topics LDA (0 = auto)")
    p.add_argument("--min-cooc", type=int, default=CFG.min_cooc)
    p.add_argument("--viz-nodes", type=int, default=CFG.graph_viz_nodes)
    p.add_argument("--offline-plotly", action="store_true", help="embarque plotly.js (rapport 100 %% hors-ligne, +4 Mo)")
    p.add_argument("--debug", action="store_true", help="affiche les traces d'erreur complètes")
    return p.parse_args(argv)


def main(argv=None):
    global DEBUG
    a = parser_arguments(argv)
    DEBUG = a.debug
    if a.repos_file:
        CFG.repos = [l.strip() for l in Path(a.repos_file).read_text(encoding="utf-8").splitlines()
                     if l.strip() and not l.strip().startswith("#")]
    CFG.output_dir, CFG.work_dir = Path(a.out), Path(a.work)
    CFG.force, CFG.cache_ttl_h, CFG.history, CFG.workers = a.force, a.ttl, not a.no_history, a.workers
    CFG.embeddings, CFG.embed_model = not a.no_embeddings, a.embed_model
    CFG.max_passages, CFG.max_embed, CFG.n_procs = a.max_passages, a.max_embed, a.procs
    CFG.n_topics, CFG.min_cooc, CFG.graph_viz_nodes, CFG.offline_plotly = a.topics, a.min_cooc, a.viz_nodes, a.offline_plotly
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    debut = time.time()

    banner("PROTOCOLE D'EXTRACTION : LE GRIMOIRE DU GLITCH", "Version 3.0 — Text Mining & DataViz",
           couleur=C.BMAGENTA, symbole="✦")
    print(f"{C.DIM}  Invocation lancée le {datetime.now():%Y-%m-%d %H:%M:%S}{C.RESET}")
    print(f"{C.DIM}  Python {sys.version.split()[0]}  │  Cwd : {Path.cwd()}{C.RESET}")
    flag = lambda b: "✓" if b else "✗"
    print(f"{C.DIM}  Modules : spaCy={flag(HAS_SPACY)} │ sentence-transformers={flag(HAS_ST)} │ "
          f"wordcloud={flag(HAS_WORDCLOUD)} │ pyvis={flag(HAS_PYVIS)} │ plotly={flag(HAS_PLOTLY)} │ "
          f"umap={flag(HAS_UMAP)} │ adjustText={flag(HAS_ADJUST)}{C.RESET}")

    repos, cdf = initialiser_sanctuaire()
    if not repos:
        log("Aucun dépôt disponible. Vérifiez le réseau / les URLs.", "err")
        return 1
    cor = moissonner_la_substance(repos)
    if sum(len(d["texte"]) for d in cor.docs) < 1000:
        log("Substrat mémétique trop faible. Vérifiez les dépôts.", "err")
        return 1

    A, L = analyser_le_noyau_non_lineaire(cor, repos, cdf)
    A["rep_sel"] = (A.get("sem") or {}).get("sel", [])

    banner("PHASE III-c — TISSAGE & ANALYSE DU GRAPHE", "Co-occurrences, centralités, communautés",
           couleur=C.BBLUE, symbole="❖")
    G = safe("graphe", construire_graphe, L, default=nx.Graph())
    ag, G = safe("analyse graphe", analyser_graphe, G, default=({}, G))
    ag = ag or _ag_vide()
    A["rep_sel"] = (A.get("sem") or {}).get("sel", [])

    grimoire = safe("grimoire", forger_le_grimoire, A, ag, G, cor, repos, cdf, default=None)

    banner("PHASE V — PROJECTION DE LA MOUSSE QUANTIQUE", "DataViz avancée", couleur=C.BCYAN, symbole="❖")
    if ag.get("pagerank"):
        safe("viz mousse", viz_mousse_quantique, G, ag)
        safe("viz ego", viz_ego_networks, G, ag)
        safe("viz heatmap", viz_heatmap_centralites, G, ag)
        safe("viz communautés", viz_communautes_matrice, ag)
        safe("pyvis", generer_html_interactif, G, ag, A)
    safe("viz wordclouds", viz_wordclouds, A)
    safe("viz distinctifs", viz_distinctifs_barres, A)
    safe("viz topics", viz_topics, A)
    safe("viz carte", viz_carte_semantique, A)
    safe("viz thèmes×dépôts", viz_themes_x_depots, A)
    safe("viz similarité", viz_similarite_depots, A)
    safe("viz stylométrie", viz_stylometrie, A)
    safe("viz zipf", viz_zipf, A)
    safe("viz ner", viz_ner, A)
    safe("viz inventaire", viz_inventaire, cor)
    safe("viz temporalité", viz_temporalite, A)
    safe("viz références", viz_references, cor)
    safe("viz code", viz_code, cor)
    if grimoire:
        safe("rapport html", generer_rapport_html, A, ag, G, cor, cdf, grimoire, time.time() - debut)

    duree = time.time() - debut
    safe("manifeste", ecrire_manifeste, repos, cor, duree)
    afficher_synthese_finale(cor, A, ag, duree)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{C.BRED}⚠ Rituel interrompu par l'opérateur.{C.RESET}\n")
        sys.exit(1)
