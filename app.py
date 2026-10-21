"""Rota Segura - Flask + OpenStreetMap (Nominatim/OSRM) + risco de acidentes (PRF) + Machine Learning."""
import glob, gzip, hashlib, json, os, time
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import joblib, sklearn

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OSRM_URL = os.getenv("OSRM_URL", "https://router.project-osrm.org").rstrip("/")   # troque por um OSRM local se quiser
NOMINATIM_EMAIL = os.getenv("NOMINATIM_EMAIL", "")                                   # opcional, boa prática da política do Nominatim
UA = "RotaSegura-PI-UNIVESP/1.0" + (f" ({NOMINATIM_EMAIL})" if NOMINATIM_EMAIL else "")
RAIO_M = int(os.getenv("RAIO_CORREDOR_M", "150"))
PEDAGIO_RAIO_M = int(os.getenv("PEDAGIO_RAIO_M", "500"))   # praça conta se estiver a ate X m da rota
LIM_BAIXO, LIM_ALTO = 0.8, 1.5          # razao vs. media nacional (UPS/km)
CATS = ["uf", "br", "dia_semana", "fase_dia", "condicao_metereologica", "tipo_pista"]
NUMS = ["hora", "mes", "km"]
DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]


# ---------------------------------------------------------------- dados
def carregar(pasta="data"):
    arqs = sorted(glob.glob(os.path.join(pasta, "datatran*.csv")))
    if not arqs:
        raise SystemExit("Coloque os CSVs datatran*.csv da PRF na pasta data/")
    dfs = []
    for a in arqs:
        try:
            dfs.append(pd.read_csv(a, sep=";", encoding="utf-8", decimal=",", low_memory=False))
        except UnicodeDecodeError:
            dfs.append(pd.read_csv(a, sep=";", encoding="latin1", decimal=",", low_memory=False))
    d = pd.concat(dfs, ignore_index=True)
    d.columns = [c.strip().lower() for c in d.columns]
    for c in ("latitude", "longitude", "km"):
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "."), errors="coerce")
    d = d.dropna(subset=["latitude", "longitude", "km", "br", "data_inversa"]).copy()
    d = d[d.latitude.between(-34, 6) & d.longitude.between(-74, -28)]
    for c in ("mortos", "feridos_leves", "feridos_graves", "feridos"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    d["data"] = pd.to_datetime(d.data_inversa, errors="coerce")
    d = d.dropna(subset=["data"])
    d["hora"] = pd.to_numeric(d.horario.astype(str).str[:2], errors="coerce").fillna(0).astype(int)
    d["mes"] = d.data.dt.month
    d["br"] = d.br.astype(int).astype(str)
    d["grave"] = ((d.mortos > 0) | (d.feridos_graves > 0) | (d.classificacao_acidente == "Com Vítimas Fatais")).astype(int)
    d["ups"] = np.where(d.mortos > 0, 13, np.where(d.feridos > 0, 4, 1))   # UPS (DNIT): 1 / 4 / 13
    if "municipio" not in d:
        d["municipio"] = ""
    d["municipio"] = d.municipio.fillna("").astype(str)
    for c in CATS:
        d[c] = d[c].fillna("NA").astype(str)
    keep = CATS + NUMS + ["latitude", "longitude", "mortos", "feridos", "feridos_graves", "data", "grave", "ups"]
    return d[list(dict.fromkeys(keep))].reset_index(drop=True)


MODEL_VERSION = 4


def treinar(d):
    """Compara 3 modelos (corte temporal 80/20, com amostragem para bases grandes) e devolve o melhor."""
    rng = np.random.RandomState(0)
    X, y = d[CATS + NUMS], d.grave.values
    ante = np.flatnonzero((d.data <= d.data.quantile(0.8)).values)
    post = np.flatnonzero((d.data > d.data.quantile(0.8)).values)
    tr = rng.choice(ante, min(len(ante), 150_000), replace=False)
    te = rng.choice(post, min(len(post), 60_000), replace=False)
    oh = lambda: ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), CATS), ("n", StandardScaler(), NUMS)])
    od = lambda: ColumnTransformer([("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32), CATS)],
                                   remainder="passthrough")
    mask = [True] * len(CATS) + [False] * len(NUMS)
    cand = {
        "Regressão Logística": lambda: Pipeline([("pre", oh()), ("m", LogisticRegression(max_iter=300))]),
        "Random Forest": lambda: Pipeline([("pre", oh()), ("m", RandomForestClassifier(80, min_samples_leaf=10, max_samples=0.5, n_jobs=-1, random_state=0))]),
        "Gradient Boosting": lambda: Pipeline([("pre", od()), ("m", HistGradientBoostingClassifier(categorical_features=mask, random_state=0))]),
    }
    auc = {}
    print("Treinando e comparando modelos (so na primeira vez)...")
    for nome, mk in cand.items():
        t0 = time.time()
        p = mk().fit(X.iloc[tr], y[tr])
        auc[nome] = round(float(roc_auc_score(y[te], p.predict_proba(X.iloc[te])[:, 1])), 4)
        print(f"  {nome}: AUC {auc[nome]} ({time.time() - t0:.0f}s)")
    melhor = max(auc, key=auc.get)
    todos = rng.choice(len(d), min(len(d), 250_000), replace=False)
    final = cand[melhor]().fit(X.iloc[todos], y[todos])
    info = {"auc_teste": auc, "escolhido": melhor, "taxa_base_graves": round(float(y.mean()), 4),
            "registros": int(len(d)), "periodo": [str(d.data.min().date()), str(d.data.max().date())]}
    return final, info


def modelo_com_cache(d, pasta="data"):
    """Treina uma vez e guarda em .cache/; so retreina se os CSVs mudarem."""
    arqs = sorted(glob.glob(os.path.join(pasta, "datatran*.csv")))
    sig = hashlib.md5(json.dumps([[os.path.basename(a), os.path.getsize(a), os.stat(a).st_mtime_ns] for a in arqs]
                                 + [MODEL_VERSION, sklearn.__version__]).encode()).hexdigest()[:12]
    arq = os.path.join(".cache", f"modelo_{sig}.joblib")
    if os.path.exists(arq):
        try:
            print("Usando modelo em cache:", arq)
            return joblib.load(arq)
        except Exception:
            pass
    final, info = treinar(d)
    X = d[CATS + NUMS]
    p = np.concatenate([final.predict_proba(X.iloc[i:i + 100_000])[:, 1] for i in range(0, len(X), 100_000)])
    os.makedirs(".cache", exist_ok=True)
    joblib.dump((final, info, p), arq)
    return final, info, p


print("Carregando dados da PRF...")
D = carregar()
print(f"{len(D)} acidentes. Preparando modelo (a 1a vez demora; depois fica em cache)...")
MODELO, INFO, _P = modelo_com_cache(D)
D["p"] = _P                                              # P(grave) por acidente, usada no painel
print("Modelos (AUC no teste):", INFO["auc_teste"], "->", INFO["escolhido"])
TREE = BallTree(np.radians(D[["latitude", "longitude"]].values), metric="haversine")
UPS_KM_NACIONAL = D.ups.sum() / D.groupby(["br", "uf"]).km.max().clip(lower=10).sum()   # media por km de BR/UF (km reinicia por UF)


# ---------------------------------------------------------------- pedagios (ANTT)
CATEGORIAS = {"moto": "Moto", "carro": "Carro", "caminhao_2_eixos": "Caminhão 2 eixos", "caminhao_3_eixos": "Caminhão 3 eixos",
              "caminhao_4_eixos": "Caminhão 4 eixos", "caminhao_6_eixos": "Caminhão 6 eixos"}


def carregar_pedagios(pasta="data"):
    """CSV long (id,nome,rodovia,uf,latitude,longitude,categoria,preco) -> 1 linha por praca, 1 coluna por categoria."""
    arq = os.path.join(pasta, "pedagios_federais_antt.csv")
    if not os.path.exists(arq):
        return None
    try:
        p = pd.read_csv(arq, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        p = pd.read_csv(arq, sep=None, engine="python", encoding="latin1")
    p.columns = [c.strip().lower() for c in p.columns]
    p["categoria"] = p.categoria.astype(str).str.strip().str.lower()
    p["preco"] = pd.to_numeric(p.preco.astype(str).str.replace(",", "."), errors="coerce")
    p = p.dropna(subset=["latitude", "longitude", "preco"])
    return p.pivot_table(index=["nome", "rodovia", "uf", "latitude", "longitude"], columns="categoria", values="preco", aggfunc="first").reset_index()


PEDAGIOS = carregar_pedagios()
PED_TREE = BallTree(np.radians(PEDAGIOS[["latitude", "longitude"]].values), metric="haversine") if PEDAGIOS is not None else None
PED_UFS = sorted(PEDAGIOS.uf.unique()) if PEDAGIOS is not None else []
print(f"Pedágios: {len(PEDAGIOS)} praças ({', '.join(PED_UFS)})" if PEDAGIOS is not None else "Pedágios: arquivo data/pedagios_federais_antt.csv não encontrado (recurso desativado)")


# ---------------------------------------------------------------- geometria
def decodificar(enc):
    pts, i, lat, lng = [], 0, 0, 0
    while i < len(enc):
        for eixo in (0, 1):
            r = s = 0
            while True:
                b = ord(enc[i]) - 63; i += 1
                r |= (b & 31) << s; s += 5
                if b < 32:
                    break
            v = ~(r >> 1) if r & 1 else r >> 1
            if eixo == 0: lat += v
            else: lng += v
        pts.append((lat / 1e5, lng / 1e5))
    return np.array(pts)


def adensificar(p, passo_m=60):
    if len(p) < 2:
        return p
    dl, dg = np.diff(p[:, 0]) * 111320, np.diff(p[:, 1]) * 111320 * np.cos(np.radians(p[:-1, 0]))
    n = np.maximum(1, (np.hypot(dl, dg) // passo_m).astype(int))
    idx = np.repeat(np.arange(len(n)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 1) / np.repeat(n, n)
    return np.vstack([p[:1], p[idx] + (p[idx + 1] - p[idx]) * t[:, None]])


def pedagios_na_rota(pts, categoria):
    """Pracas a ate PEDAGIO_RAIO_M da rota (na ordem do percurso) e total para a categoria de veiculo."""
    if PEDAGIOS is None:
        return None
    primeiro = {}
    for i, achados in enumerate(PED_TREE.query_radius(np.radians(pts), PEDAGIO_RAIO_M / 6371000.0)):
        for j in achados:
            primeiro.setdefault(int(j), i)
    pracas = []
    for j in sorted(primeiro, key=primeiro.get):
        r = PEDAGIOS.iloc[j]
        v = r.get(categoria)
        pracas.append({"nome": r.nome, "rodovia": r.rodovia, "uf": r.uf, "lat": float(r.latitude), "lon": float(r.longitude),
                       "preco": None if v is None or pd.isna(v) else round(float(v), 2)})
    return {"categoria": categoria, "n": len(pracas), "total": round(sum(p["preco"] or 0 for p in pracas), 2), "pracas": pracas}


def fase_dia(h):
    return "Amanhecer" if h in (5,) else "Pleno dia" if 6 <= h < 18 else "Anoitecer" if h == 18 else "Plena Noite"


@lru_cache(maxsize=256)
def p_nacional(hora, dia, mes):
    return float(_prever(D.sample(min(4000, len(D)), random_state=0), hora, dia, mes).mean())


def _prever(m, hora, dia, mes):
    X = m[CATS + NUMS].copy()
    X["hora"], X["mes"], X["dia_semana"] = hora, mes, DIAS[dia]
    X["fase_dia"], X["condicao_metereologica"] = fase_dia(hora), "Céu Claro"
    return MODELO.predict_proba(X)[:, 1]


def analisar(poly, dist_km, hora, dia, mes):
    pts = adensificar(decodificar(poly))
    achados = TREE.query_radius(np.radians(pts), RAIO_M / 6371000.0)
    achados = [a for a in achados if len(a)]
    m = D.iloc[np.unique(np.concatenate(achados))] if achados else D.iloc[0:0]
    razao = (m.ups.sum() / max(dist_km, 1)) / UPS_KM_NACIONAL
    classe = "Baixo" if razao <= LIM_BAIXO else "Médio" if razao <= LIM_ALTO else "Alto"
    crit = []
    if len(m):
        g = m.assign(faixa=(m.km // 25 * 25).astype(int)).groupby(["br", "uf", "faixa"])
        t = g.agg(n=("ups", "size"), mortes=("mortos", "sum"), ups=("ups", "sum"), k0=("km", "min"), k1=("km", "max"))
        for (br, uf, _), r in t[t.n >= 5].sort_values("ups", ascending=False).head(3).iterrows():
            crit.append({"br": br, "uf": uf, "km_ini": int(r.k0), "km_fim": max(int(np.ceil(r.k1)), int(r.k0) + 1),
                         "acidentes": int(r.n), "mortes": int(r.mortes)})
    p = float(_prever(m, hora, dia, mes).mean()) if len(m) else None
    graves = m[m.grave == 1].head(400)
    return {"acidentes": int(len(m)), "acidentes_com_mortes": int((m.mortos > 0).sum()), "mortos": int(m.mortos.sum()),
            "feridos": int(m.feridos.sum()), "ups": int(m.ups.sum()),
            "acidentes_100km": round(len(m) / max(dist_km, 1) * 100, 1),
            "razao_media_nacional": round(float(razao), 2), "classe": classe,
            "p_grave_modelo": None if p is None else round(p, 4), "p_grave_nacional": round(p_nacional(hora, dia, mes), 4),
            "trechos_criticos": crit, "graves": graves[["latitude", "longitude"]].round(5).values.tolist()}


# ---------------------------------------------------------------- rotas (OpenStreetMap)
class ErroRota(Exception):
    def __init__(self, msg, status=502):
        super().__init__(msg); self.status = status


@lru_cache(maxsize=512)
def geocodificar(texto):
    """Nominatim (OSM): 1 consulta/segundo e User-Agent identificado, conforme a politica de uso."""
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search", headers={"User-Agent": UA}, timeout=15,
                         params={"q": texto, "format": "jsonv2", "limit": 1, "countrycodes": "br"})
        r.raise_for_status()
        j = r.json()
    except (requests.RequestException, ValueError) as e:
        raise ErroRota(f"Falha ao buscar o endereço '{texto}': {e}")
    if not j:
        raise ErroRota(f"Endereço não encontrado: {texto}", 404)
    time.sleep(1.0)
    return float(j[0]["lat"]), float(j[0]["lon"])


def osrm_rotas(origem, destino):
    (la1, lo1), (la2, lo2) = geocodificar(origem), geocodificar(destino)
    try:
        r = requests.get(f"{OSRM_URL}/route/v1/driving/{lo1},{la1};{lo2},{la2}", headers={"User-Agent": UA}, timeout=60,
                         params={"alternatives": 3, "overview": "full", "geometries": "polyline", "steps": "false"})
        j = r.json()
    except (requests.RequestException, ValueError) as e:
        raise ErroRota(f"Falha ao consultar o OSRM: {e}")
    if r.status_code != 200 or j.get("code") != "Ok":
        raise ErroRota("OSRM: " + str(j.get("message") or j.get("code") or r.status_code), 404 if j.get("code") == "NoRoute" else 502)
    return [{"distancia_km": round(x["distance"] / 1000, 1), "duracao_min": round(x["duration"] / 60), "polyline": x["geometry"]}
            for x in j["routes"]]


# ---------------------------------------------------------------- web
app = Flask(__name__)


def _fmt(n):
    return f"{int(n):,}".replace(",", ".")


STATS = {"registros": _fmt(len(D)), "graves": _fmt(D.grave.sum()), "mortos": _fmt(D.mortos.sum()), "brs": int(D.br.nunique()),
         "ufs": int(D.uf.nunique()), "auc": str(INFO["auc_teste"][INFO["escolhido"]]).replace(".", ","), "modelo": INFO["escolhido"],
         "pracas": 0 if PEDAGIOS is None else len(PEDAGIOS), "ini": pd.Timestamp(INFO["periodo"][0]).strftime("%d/%m/%Y"), "fim": pd.Timestamp(INFO["periodo"][1]).strftime("%d/%m/%Y")}


@app.get("/")
def inicio():
    return render_template("landing.html", s=STATS)


@app.get("/rota")
def rota():
    return render_template("rota.html", s=STATS, info=INFO, categorias=CATEGORIAS, ped_ativo=PEDAGIOS is not None, ped_ufs=", ".join(PED_UFS))


@app.post("/api/rota")
def api_rota():
    j = request.get_json(force=True, silent=True) or {}
    o, d = (j.get("origem") or "").strip(), (j.get("destino") or "").strip()
    if not o or not d:
        return jsonify(erro="Informe origem e destino."), 400
    saida = None
    if j.get("saida"):
        try:
            saida = datetime.fromisoformat(j["saida"])
        except ValueError:
            return jsonify(erro="Data/hora de saída inválida."), 400
    ref = saida or datetime.now()
    cat = j.get("categoria") or "carro"
    if cat not in CATEGORIAS:
        return jsonify(erro="Categoria de veículo inválida."), 400
    try:
        rotas = osrm_rotas(o, d)
    except ErroRota as e:
        return jsonify(erro=str(e)), e.status
    if not rotas:
        return jsonify(erro="Nenhuma rota encontrada."), 404
    for r in rotas:
        r["risco"] = analisar(r["polyline"], r["distancia_km"], ref.hour, ref.weekday(), ref.month)
        r["pedagio"] = pedagios_na_rota(adensificar(decodificar(r["polyline"])), cat)
    rapida = min(rotas, key=lambda r: r["duracao_min"])
    curta = min(rotas, key=lambda r: r["distancia_km"])
    segura = min(rotas, key=lambda r: r["risco"]["razao_media_nacional"])
    totais = {r["pedagio"]["total"] for r in rotas if r["pedagio"]}
    barata = min((r for r in rotas if r["pedagio"]), key=lambda r: r["pedagio"]["total"], default=None) if len(totais) > 1 else None
    for r in rotas:
        r["rotulos"] = [t for t, c in (("mais rápida", r is rapida), ("mais curta", r is curta), ("menor risco", r is segura),
                                      ("menor pedágio", r is barata)) if c]
    return jsonify(rotas=rotas, referencia=ref.isoformat(timespec="minutes"))


@app.get("/painel")
def painel():
    return render_template("painel.html", s=STATS)


@lru_cache(maxsize=1)
def _dados_gz():
    d = D
    brs = sorted(d.br.unique(), key=int)
    ufs = sorted(d.uf.unique())
    mo = d.data.dt.year * 12 + d.data.dt.month - 1
    cols = {"brs": brs, "ufs": ufs, "br": pd.Categorical(d.br, categories=brs).codes.tolist(),
            "uf": pd.Categorical(d.uf, categories=ufs).codes.tolist(), "km": d.km.round(1).tolist(),
            "bin": np.minimum(5, d.hora // 4).tolist(), "dow": ((d.data.dt.dayofweek + 1) % 7).tolist(),
            "mo0": int(mo.min()), "mo": (mo - mo.min()).tolist(), "g": d.grave.tolist(), "m": (d.mortos > 0).astype(int).tolist(),
            "fe": d.feridos.astype(int).tolist(), "lat": d.latitude.round(3).tolist(), "lon": d.longitude.round(3).tolist(),
            "p": d.p.round(3).tolist(), "info": INFO, "fonte": f"PRF datatran ({INFO['periodo'][0]} a {INFO['periodo'][1]})"}
    return gzip.compress(json.dumps(cols, separators=(",", ":")).encode(), 6)


@app.get("/api/dados")
def api_dados():
    return Response(_dados_gz(), mimetype="application/json", headers={"Content-Encoding": "gzip"})


@app.get("/api/modelo")
def api_modelo():
    return jsonify(INFO)


@app.get("/api/previsoes.csv")
def previsoes():
    """CSV br;uf;km;prob por trecho de 10 km - pode ser importado no painel HTML."""
    t = pd.DataFrame({"br": D.br, "uf": D.uf, "km": (D.km // 10 * 10).astype(int), "prob": D.p})
    t = t.groupby(["br", "uf", "km"], as_index=False).agg(prob=("prob", "mean"), n=("prob", "size"))
    return Response(t.to_csv(sep=";", index=False, decimal=","), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=previsoes_trechos.csv"})


if __name__ == "__main__":
    app.run(port=5000)
