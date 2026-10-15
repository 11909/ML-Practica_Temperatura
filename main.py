"""
Predicción de temperatura - Edificio de Gobierno de la ESCOM-IPN, Zacatenco, CDMX
Objetivo: temperatura del aire el día OBJETIVO (15 de octubre de 2026) a las 07:00, 07:30, 08:00 y 08:30
en el punto 19.505277, -99.146525 (2240 m), a ~0.8 km de la EMA ENCB II.
Meteostat no tiene estación en la ESCOM: sus estaciones más cercanas (Aeropuerto, Tacubaya)
se usan sólo como fuente de historia larga, y el sesgo local (EMA ENCB II) traslada el
pronóstico de esa fuente al sitio.

Idea general (ver README.md):
    pronóstico = climatología(fecha, hora)      <- regresión entrenada con años de datos horarios
               + alfa * anomalía reciente      <- cuánto "recuerda" la atmósfera las últimas semanas
               + sesgo local del sitio         <- diferencia EMA - fuente, en función del sol
"""
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meteostat as ms
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "outputs"
EMA_CSV = BASE / "Dataset_EMA_ENCB.csv"

# Punto donde se quiere el pronóstico: Edificio de Gobierno de la ESCOM-IPN
SITIO = "Edificio de Gobierno ESCOM"
LAT, LON, ELEV = 19.505276995427376, -99.14652518924372, 2240
# La CDMX no tiene horario de verano desde 2022. Para que "07:00" signifique lo mismo
# en todos los años, todo se convierte a UTC-6 fijo.
UTC_OFFSET = pd.Timedelta(hours=-6)
MS_START = 2017            # Meteostat tiene datos horarios completos desde 2017
POWER_START = 2005         # NASA POWER es completo desde 2001
OBJETIVO = pd.Timestamp("2026-10-15")   # día a pronosticar (sólo hay que cambiar esta línea)
HORAS = ["07:00", "07:30", "08:00", "08:30"]
VENTANA_ANOM = 21          # días usados para medir la anomalía reciente
HORAS_MANANA = (6.0, 9.0)  # franja horaria relevante (horas decimales)
ANIOS_VALIDACION = range(2018, 2026)
VENTANA_VAL = 7            # la validación usa los días OBJETIVO ± VENTANA_VAL de cada año
# Fracción del sesgo local (EMA - Meteostat) que se aplica al pronóstico: 1 = todo, 0 = nada.
# El sesgo sólo pudo medirse con la EMA de jul-ago 2026; ver incertidumbre_traslado().
PESO_SESGO = 1.0
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def etiqueta(fecha):
    return f"{fecha.day}-{MESES[fecha.month - 1]}"


ETQ = etiqueta(OBJETIVO)
ETQ_VAL = (f"{etiqueta(OBJETIVO - pd.Timedelta(days=VENTANA_VAL))} a "
           f"{etiqueta(OBJETIVO + pd.Timedelta(days=VENTANA_VAL))}")


# ----------------------------------------------------------------------------- datos
def load_ema():
    """EMA ENCB II: registros cada 10 min en orden inverso. Se usa la hora UTC -> UTC-6 y se
    eliminan valores imposibles y picos (saltos que aparecen sobre todo después de huecos)."""
    meta = {}
    with open(EMA_CSV, encoding="utf-8") as f:
        for linea in list(f)[:9]:
            partes = linea.strip().split(",", 1)
            if len(partes) == 2:
                meta[partes[0]] = partes[1].strip('"')
    df = pd.read_csv(EMA_CSV, skiprows=9, encoding="utf-8")
    t = pd.to_datetime(df["Fecha UTC"]) + UTC_OFFSET
    ema = pd.Series(df["Temperatura del Aire (°C)"].values, index=t, name="temp")
    ema = ema[~ema.index.duplicated()].sort_index().dropna()
    n0 = len(ema)
    ema = ema[(ema > -5) & (ema < 40)]
    mediana = ema.rolling("50min", center=True).median()
    ema = ema[(ema - mediana).abs() <= 3]
    print(f"EMA {meta.get('Estación')} ({meta.get('Latitud')}, {meta.get('Longitud')}, "
          f"{meta.get('Altitud')} m): {n0 - len(ema)} valores descartados por control de calidad")
    return ema


def _hourly_estacion(sid, inicio, fin):
    partes = []
    for anio in range(inicio, fin.year + 1):
        df = ms.hourly(sid, datetime(anio, 1, 1), min(datetime(anio, 12, 31, 23), fin),
                       parameters=[ms.Parameter.TEMP]).fetch()
        if df is not None:
            s = df["temp"]
            if isinstance(s.index, pd.MultiIndex):
                s = s.droplevel(0)
            partes.append(s)
    s = pd.concat(partes).astype(float).dropna()
    s.index = pd.to_datetime(s.index) + UTC_OFFSET
    return s[~s.index.duplicated()].sort_index()


def _leer_cache(cache, anio_inicio):
    """Devuelve (serie guardada o None, año desde el que hay que descargar). Si la caché ya
    existe sólo se vuelve a bajar desde el año de su último dato, para tener lo más reciente
    posible sin repetir toda la descarga. Si se actualizó hace menos de 1 h se usa tal cual."""
    if not cache.exists():
        return None, anio_inicio
    vieja = pd.read_csv(cache, index_col=0, parse_dates=True)["temp"]
    edad = datetime.now().timestamp() - cache.stat().st_mtime
    if edad < 3600:
        return vieja, None
    return vieja, vieja.index.max().year


def _guardar_cache(cache, nueva, vieja, nombre):
    s = nueva if vieja is None else nueva.combine_first(vieja)
    s = s[~s.index.duplicated()].sort_index()
    if vieja is not None:
        print(f"  {nombre}: caché actualizada hasta {s.index.max()} "
              f"({len(s) - len(vieja)} h nuevas)")
    DATA.mkdir(exist_ok=True)
    s.to_frame("temp").to_csv(cache)
    return s


def fetch_meteostat():
    """Estación principal = la más cercana a la ESCOM (Aeropuerto, 10 km, 2234 m: mismo fondo
    plano del valle que Zacatenco). No es el sitio objetivo: su diferencia con la ESCOM se
    corrige después con el sesgo local medido en la EMA ENCB II. Sus huecos se rellenan con
    la segunda más cercana (Tacubaya), corregida por la diferencia media entre ambas a cada
    hora del día."""
    cache = DATA / f"meteostat_{LAT:.3f}_{LON:.3f}.csv"
    vieja, desde = _leer_cache(cache, MS_START)
    if desde is None:
        return vieja
    punto = ms.Point(LAT, LON, ELEV)
    est = ms.stations.nearby(punto, radius=30000, limit=2)
    print("Estaciones Meteostat:", " | ".join(f"{n} ({d / 1000:.1f} km)"
                                              for n, d in zip(est["name"], est["distance"])))
    principal, respaldo = (_hourly_estacion(sid, desde, datetime.now()) for sid in est.index)
    comun = principal.index.intersection(respaldo.index)
    dif = principal[comun] - respaldo[comun]
    ajuste = dif.groupby(dif.index.hour).mean()
    respaldo_aj = respaldo + ajuste.reindex(respaldo.index.hour).values
    s = principal.combine_first(respaldo_aj)
    print(f"  Meteostat: {len(principal)} h de la principal, {len(s) - len(principal)} h rellenadas")
    return _guardar_cache(cache, s, vieja, "Meteostat")


def fetch_power():
    """NASA POWER (MERRA-2, celda ~50 km, altitud de la celda ~2600 m), T2M horaria en UTC."""
    cache = DATA / f"power_{LAT:.3f}_{LON:.3f}.csv"
    vieja, desde = _leer_cache(cache, POWER_START)
    if desde is None:
        return vieja
    partes = []
    for anio in range(desde, datetime.now().year + 1):
        r = requests.get(
            "https://power.larc.nasa.gov/api/temporal/hourly/point",
            params={"parameters": "T2M", "community": "RE", "latitude": LAT, "longitude": LON,
                    "start": f"{anio}0101", "end": f"{anio}1231", "format": "JSON",
                    "time-standard": "UTC"},
            timeout=180)
        r.raise_for_status()
        valores = r.json()["properties"]["parameter"]["T2M"]
        partes.append(pd.Series(valores))
        print(f"  POWER {anio}: {len(valores)} horas")
    s = pd.concat(partes).astype(float)
    s.index = pd.to_datetime(s.index, format="%Y%m%d%H") + UTC_OFFSET
    s = s[s > -900].sort_index()  # -999 = dato faltante
    return _guardar_cache(cache, s, vieja, "NASA POWER")


# ----------------------------------------------------------------------------- utilidades
def a_30min(s):
    """Lleva una serie horaria a una rejilla de 30 min (interpola sólo huecos de 1 paso)."""
    return s.resample("30min").mean().interpolate(limit=1, limit_area="inside")


def hora_decimal(idx):
    return idx.hour + idx.minute / 60


def es_manana(idx):
    h = hora_decimal(idx)
    return (h >= HORAS_MANANA[0]) & (h <= HORAS_MANANA[1])


def tiempos_objetivo(dia):
    return pd.DatetimeIndex([pd.Timestamp(f"{dia.date()} {h}") for h in HORAS])


def elevacion_solar(idx, lat=LAT, lon=LON):
    """Altura del sol (grados) con las fórmulas de la NOAA. idx en hora local UTC-6."""
    utc = pd.DatetimeIndex(idx) - UTC_OFFSET
    h = hora_decimal(utc)
    g = 2 * np.pi / 365 * (utc.dayofyear - 1 + (h - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ecuacion = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                         - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    tiempo_solar = h * 60 + ecuacion + 4 * lon
    angulo = np.radians(tiempo_solar / 4 - 180)
    phi = np.radians(lat)
    sen = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(angulo)
    return np.degrees(np.arcsin(np.asarray(sen)))


# ----------------------------------------------------------------------------- modelo
def build_features(idx):
    """Variables cíclicas: con seno/coseno el 31-dic queda junto al 1-ene y las 23:00 junto a
    las 00:00. Los productos día×hora permiten que el ciclo diario cambie con la estación
    (en octubre amanece más tarde y la amplitud térmica es mayor que en julio)."""
    idx = pd.DatetimeIndex(idx)
    h = hora_decimal(idx)
    doy = idx.dayofyear + h / 24
    a = 2 * np.pi * doy / 365.25
    b = 2 * np.pi * h / 24
    X = {"tendencia": (idx.year + doy / 365.25 - 2015) / 10}  # calentamiento / urbanización
    est = {f"{f}{k}": fn(k * a) for k in (1, 2) for f, fn in (("ds", np.sin), ("dc", np.cos))}
    dia = {f"{f}{k}": fn(k * b) for k in (1, 2, 3) for f, fn in (("hs", np.sin), ("hc", np.cos))}
    X.update(est)
    X.update(dia)
    for ke, ve in est.items():
        for kd, vd in dia.items():
            X[f"{ke}x{kd}"] = ve * vd
    return pd.DataFrame(X, index=idx)


def nuevo_modelo(tipo):
    if tipo == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         random_state=0)


class Pronosticador:
    """Climatología (regresión) + persistencia de la anomalía, para una fuente horaria dada."""

    def __init__(self, serie, tipo="ridge", excluir_anio=None):
        entren = serie if excluir_anio is None else serie[serie.index.year != excluir_anio]
        self.modelo = nuevo_modelo(tipo).fit(build_features(entren.index), entren.values)
        # Anomalías de mañana (promedio diario) sobre toda la serie disponible
        s30 = a_30min(serie).dropna()
        anom = s30 - self.climatologia(s30.index)
        manana = anom[es_manana(anom.index)]
        self.anom_diaria = manana.groupby(manana.index.normalize()).mean()
        self.alfa = None
        self.entren_anios = set(entren.index.year)

    def climatologia(self, idx):
        return pd.Series(self.modelo.predict(build_features(idx)), index=idx)

    def anomalia_reciente(self, emision):
        ini = emision - pd.Timedelta(days=VENTANA_ANOM - 1)
        v = self.anom_diaria.loc[ini:emision]
        return v.mean() if len(v) >= VENTANA_ANOM // 2 else np.nan

    def ajustar_alfa(self, plazo_dias):
        """Regresión (sin intercepto) de la anomalía en el día objetivo sobre la anomalía de las
        3 semanas previas a la emisión, usando sólo años de entrenamiento y sep-dic."""
        xs, ys = [], []
        for dia, y in self.anom_diaria.items():
            if dia.year not in self.entren_anios or not (9 <= dia.month <= 12):
                continue
            x = self.anomalia_reciente(dia - pd.Timedelta(days=plazo_dias))
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
        xs, ys = np.array(xs), np.array(ys)
        self.alfa = float(np.clip((xs * ys).sum() / (xs * xs).sum(), 0, 1))
        return self.alfa

    def pronostico(self, tiempos, emision):
        a = self.anomalia_reciente(emision)
        return self.climatologia(tiempos) + self.alfa * (0 if np.isnan(a) else a)


# ----------------------------------------------------------------------------- validación
def validar(serie, tipo, plazo_dias):
    """Validación dejando un año fuera: para cada año se entrena sin ese año y se pronostican
    las mañanas de OBJETIVO ± VENTANA_VAL días emitiendo `plazo_dias` antes, igual que ahora."""
    filas = []
    s30 = a_30min(serie)
    # Distancia (en días, circular) entre el día del año de cada dato y el del objetivo
    dist = np.abs(s30.index.dayofyear - OBJETIVO.dayofyear)
    ventana = np.minimum(dist, 365 - dist) <= VENTANA_VAL
    hd = hora_decimal(s30.index)
    for anio in ANIOS_VALIDACION:
        p = Pronosticador(serie, tipo, excluir_anio=anio)
        p.ajustar_alfa(plazo_dias)
        otros = ventana & (s30.index.year != anio)
        centro = OBJETIVO.replace(year=anio)
        for dia in pd.date_range(centro - pd.Timedelta(days=VENTANA_VAL),
                                 centro + pd.Timedelta(days=VENTANA_VAL)):
            t = tiempos_objetivo(dia)
            real = s30.reindex(t)
            if real.isna().any():
                continue
            emision = dia - pd.Timedelta(days=plazo_dias)
            pred = p.pronostico(t, emision)
            for i, ti in enumerate(t):
                h = hora_decimal(t)[i]
                # Línea base 1: promedio histórico de la ventana a esa hora (otros años)
                clim = s30[otros & (hd == h)].mean()
                # Línea base 2: persistencia = promedio de los 7 días previos a la emisión
                dias = [emision.normalize() - pd.Timedelta(days=k) + pd.Timedelta(hours=h)
                        for k in range(7)]
                pers = s30.reindex(dias).mean()
                filas.append({"modelo": tipo, "anio": anio, "tiempo": ti,
                              "real": real.iloc[i], "pred": pred.iloc[i],
                              "base_clim": clim, "base_persist": pers})
    return pd.DataFrame(filas)


def resumen_errores(v):
    out = {}
    for col, etiqueta in (("pred", "modelo"), ("base_clim", "climatologia simple"),
                          ("base_persist", "persistencia 7 d")):
        e = v[col] - v["real"]
        out[etiqueta] = {"MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()), "sesgo": e.mean()}
    return pd.DataFrame(out).T


# ----------------------------------------------------------------------------- sesgo local
def diferencias_ema(ema, serie):
    """EMA - fuente en los instantes :00 y :30 donde hay ambos datos."""
    ema30 = ema[ema.index.minute % 30 == 0]
    return (ema30 - a_30min(serie).reindex(ema30.index)).dropna()


def ajustar_sesgo(dif):
    """Sesgo del sitio en función de la altura del sol (mañanas, de 03:00 a 12:00).
    Por qué el sol y no la hora del reloj: la EMA sólo tiene julio-agosto, cuando a las 07:00
    el sol ya lleva ~50 min fuera; en octubre a las 07:00 apenas está saliendo. Si el sitio (o
    el sensor) se calienta con el sol antes que la estación de referencia, ese efecto sigue
    al sol, no al reloj. Antes del amanecer (altura < 0) el sesgo se toma constante."""
    el = elevacion_solar(dif.index)
    m = (dif.index.hour >= 3) & (dif.index.hour < 12) & (el < 45)
    x = np.clip(el[m], 0, None)
    y = dif[m].values
    coef = np.polyfit(x, y, 2)
    sd = float(np.std(y - np.polyval(coef, x)))
    return coef, sd


def incertidumbre_traslado(ms_serie, pw_serie, meses_ema):
    """El sesgo local se midió en los meses de la EMA y se aplica en el mes del objetivo.
    ¿Cuánto cambia de una temporada a otra la diferencia matutina entre dos sitios del valle?
    Se mide con un par que sí tiene años de datos (Meteostat - NASA POWER): para cada año,
    diferencia(mes objetivo) - diferencia(meses de la EMA). Devuelve su media cuadrática,
    que se suma como incertidumbre extra del sesgo."""
    comun = ms_serie.index.intersection(pw_serie.index)
    d = ms_serie[comun] - pw_serie[comun]
    d = d[es_manana(d.index)]
    tabla = d.groupby([d.index.year, d.index.month]).mean().unstack()
    meses = [mm for mm in meses_ema if mm in tabla.columns]
    if OBJETIVO.month in meses or OBJETIVO.month not in tabla.columns:
        return 0.0
    cambio = (tabla[OBJETIVO.month] - tabla[meses].mean(axis=1)).dropna()
    return float(np.sqrt((cambio ** 2).mean()))


def aplicar_sesgo(coef, tiempos):
    return np.polyval(coef, np.clip(elevacion_solar(tiempos), 0, None))


def sesgo_por_reloj(dif, tiempos):
    return np.array([dif[(dif.index.hour == t.hour) & (dif.index.minute == t.minute)].mean()
                     for t in tiempos])


def prueba_sesgo_ema(ema, serie):
    """Ajusta el sesgo con la primera parte de la EMA y lo prueba en la última (días nunca
    vistos): pred = fuente observada + sesgo, comparado con la EMA a las horas objetivo."""
    dif = diferencias_ema(ema, serie)
    dias = np.sort(dif.index.normalize().unique())
    corte = dias[int(len(dias) * 0.6)]
    entren, prueba = dif[dif.index < corte], dif[dif.index >= corte]
    coef, _ = ajustar_sesgo(entren)
    s30 = a_30min(serie)
    res = {"sin sesgo": [], "sesgo por reloj": [], "sesgo por sol": []}
    for dia in pd.date_range(corte, dif.index.max().normalize()):
        t = tiempos_objetivo(dia)
        real, fuente = ema.reindex(t), s30.reindex(t)
        ok = real.notna().values & fuente.notna().values
        if not ok.any():
            continue
        res["sin sesgo"].extend((fuente.values - real.values)[ok])
        res["sesgo por reloj"].extend((fuente.values + sesgo_por_reloj(entren, t) - real.values)[ok])
        res["sesgo por sol"].extend((fuente.values + aplicar_sesgo(coef, t) - real.values)[ok])
    tabla = pd.DataFrame({k: {"MAE": np.abs(v).mean(), "sesgo": np.mean(v), "n": len(v)}
                          for k, v in res.items()}).T
    return tabla, pd.Timestamp(corte).date()


# ----------------------------------------------------------------------------- gráficas
def graficas(ema, ms_serie, final, val, dif_ms, coef_ms):
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    s30 = a_30min(ms_serie)
    ini, fin = ema.index.min().normalize(), ema.index.max()
    ciclo_ema = ema.groupby(hora_decimal(ema.index)).mean()
    sel = s30.loc[ini:fin]
    ciclo_ms = sel.groupby(hora_decimal(sel.index)).mean()
    ax.plot(ciclo_ema.index, ciclo_ema.values, color="#2b6cb0", label="EMA ENCB II")
    ax.plot(ciclo_ms.index, ciclo_ms.values, color="#a0aec0", label="Meteostat (Aeropuerto)")
    ax.axvspan(7, 8.5, color="#ed8936", alpha=.15, label="horas objetivo")
    ax.set_title(f"Ciclo diario medio {ini.date()} a {fin.date()}")
    ax.set_xlabel("hora (UTC-6)")
    ax.set_ylabel("°C")
    ax.legend()

    ax = axes[0, 1]
    colores = ["#2b6cb0", "#3182ce", "#dd6b20", "#c05621"]
    for i, h in enumerate(HORAS):
        hh = pd.Timestamp(f"2000-01-01 {h}")
        s = s30[(s30.index.month == OBJETIVO.month) & (s30.index.day == OBJETIVO.day)
                & (s30.index.hour == hh.hour) & (s30.index.minute == hh.minute)]
        ax.plot(s.index.year, s.values, "o-", alpha=.5, color=colores[i], label=f"{h} Meteostat")
        ax.errorbar(OBJETIVO.year + 0.3 + i * 0.15, final.loc[h, "pred"],
                    yerr=final.loc[h, "pred"] - final.loc[h, "p10"], fmt="s", capsize=3,
                    color=colores[i])
    ax.set_title(f"{ETQ} en la historia y pronóstico {OBJETIVO.year} para {SITIO} (±80%)")
    ax.set_ylabel("°C")
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1, 0]
    tabla = (val.assign(e_modelo=(val.pred - val.real).abs(),
                        e_clim=(val.base_clim - val.real).abs(),
                        e_pers=(val.base_persist - val.real).abs())
             .groupby("anio")[["e_modelo", "e_clim", "e_pers"]].mean())
    tabla.plot.bar(ax=ax, color=["#2f855a", "#a0aec0", "#ed8936"])
    ax.set_title(f"Validación (año fuera): MAE por año, mañanas {ETQ_VAL}")
    ax.set_ylabel("MAE °C")
    ax.legend(["Modelo", "Climatología simple", "Persistencia 7 d"])

    ax = axes[1, 1]
    el = elevacion_solar(dif_ms.index)
    m = (dif_ms.index.hour >= 3) & (dif_ms.index.hour < 12) & (el < 45)
    ax.scatter(np.clip(el[m], -5, None), dif_ms[m], s=6, alpha=.3, color="#6b46c1")
    xs = np.linspace(0, 45, 50)
    ax.plot(xs, np.polyval(coef_ms, xs), color="k")
    ax.axhline(np.polyval(coef_ms, 0), xmin=0, xmax=.1, color="k")
    elt = elevacion_solar(tiempos_objetivo(OBJETIVO))
    ax.plot(np.clip(elt, 0, None), final["sesgo_local"], "s", color="#dd6b20",
            label=f"horas objetivo {ETQ}")
    ax.set_title("Sesgo EMA − Meteostat vs altura del sol (mañanas)")
    ax.set_xlabel("altura del sol (°)")
    ax.set_ylabel("°C")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT / "resumen.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- principal
def predecir_objetivo(serie, ema, tipo):
    emision = serie.index.max().normalize()   # último día con datos de la fuente
    plazo = (OBJETIVO - emision).days
    p = Pronosticador(serie, tipo)
    alfa = p.ajustar_alfa(plazo)
    t = tiempos_objetivo(OBJETIVO)
    clim = p.climatologia(t)
    anom = p.anomalia_reciente(emision)
    anom = 0.0 if np.isnan(anom) else anom
    dif = diferencias_ema(ema, serie)
    coef, sd = ajustar_sesgo(dif)
    sesgo = aplicar_sesgo(coef, t)
    res = pd.DataFrame({"elev_sol": elevacion_solar(t), "climatologia": clim.values,
                        "anomalia": alfa * anom, "sesgo_local": sesgo,
                        "sesgo_si_fuera_por_reloj": sesgo_por_reloj(dif, t),
                        "pred": clim.values + alfa * anom + PESO_SESGO * sesgo}, index=HORAS)
    info = {"emision": emision.date(), "plazo": plazo, "alfa": alfa, "anom_reciente": anom,
            "sd_sesgo": sd, "coef": coef, "dif": dif}
    return res, info


def main():
    pd.set_option("display.width", 140)
    pd.set_option("display.precision", 2)

    print("Cargando datos...")
    ema = load_ema()
    completas = {"Meteostat": fetch_meteostat(), "NASA POWER": fetch_power()}
    print(f"EMA: {ema.index.min()} -> {ema.index.max()} ({len(ema)} registros)")
    for k, s in completas.items():
        print(f"{k}: {s.index.min()} -> {s.index.max()} ({len(s)} horas)")
    # Se usan todos los datos hasta justo antes del día objetivo; los del propio día (si ya
    # existen) sólo sirven para comparar, nunca para pronosticar.
    fuentes = {k: s[s.index < OBJETIVO] for k, s in completas.items()}
    ms_serie = fuentes["Meteostat"]
    plazo = (OBJETIVO - ms_serie.index.max().normalize()).days
    print(f"\nPlazo del pronóstico: {plazo} días")

    # 1) Validación: elegir el tipo de regresión con la fuente principal (Meteostat)
    print(f"\nValidación dejando un año fuera (Meteostat, mañanas {ETQ_VAL}, "
          f"{ANIOS_VALIDACION.start}-{ANIOS_VALIDACION.stop - 1})")
    vals = {}
    for tipo in ("ridge", "hgb"):
        vals[tipo] = validar(ms_serie, tipo, plazo)
        print(f"\n[{tipo}]  n = {len(vals[tipo])}")
        print(resumen_errores(vals[tipo]))
    mejor = min(vals, key=lambda k: (vals[k].pred - vals[k].real).abs().mean())
    print(f"\nModelo elegido: {mejor}")
    val = vals[mejor]
    sd_modelo = (val.pred - val.real).std()

    # 2) ¿El sesgo local se transfiere a días no vistos? (prueba dentro de la EMA)
    tabla, corte = prueba_sesgo_ema(ema, ms_serie)
    print(f"\nPrueba del sesgo local en la EMA (ajuste antes de {corte}, prueba después),"
          " 07:00-08:30:")
    print(tabla)

    # 3) Pronóstico con ambas fuentes
    resultados = {}
    for nombre, serie in fuentes.items():
        res, info = predecir_objetivo(serie, ema, mejor)
        resultados[nombre] = (res, info)
        print(f"\n=== {nombre} | emisión {info['emision']} | alfa = {info['alfa']:.2f}"
              f" | anomalía 21 d = {info['anom_reciente']:+.2f} °C"
              f" | dispersión del sesgo = {info['sd_sesgo']:.2f} °C ===")
        print(res)

    # Meteostat es la fuente principal (observaciones); POWER (reanálisis de malla gruesa)
    # funciona como verificación independiente.
    res_ms, info_ms = resultados["Meteostat"]
    sd_traslado = incertidumbre_traslado(fuentes["Meteostat"], fuentes["NASA POWER"],
                                         sorted(set(ema.index.month)))
    print(f"\nIncertidumbre por trasladar el sesgo de los meses de la EMA a {ETQ}: "
          f"{sd_traslado:.2f} °C | peso del sesgo aplicado = {PESO_SESGO:.2f}")
    sd_total = np.sqrt(sd_modelo ** 2 + info_ms["sd_sesgo"] ** 2 + sd_traslado ** 2)
    final = pd.DataFrame({"pred": res_ms["pred"],
                          "p10": res_ms["pred"] - 1.2816 * sd_total,
                          "p90": res_ms["pred"] + 1.2816 * sd_total,
                          "verif_POWER": resultados["NASA POWER"][0]["pred"],
                          "sesgo_local": res_ms["sesgo_local"],
                          "sin_sesgo_local": res_ms["pred"] - PESO_SESGO * res_ms["sesgo_local"]},
                         index=HORAS)
    # Observación de Meteostat en el día objetivo, si ya está disponible
    obs = a_30min(completas["Meteostat"]).reindex(tiempos_objetivo(OBJETIVO))
    final["obs_Meteostat"] = obs.values
    print(f"\n========== PRONÓSTICO {ETQ}-{OBJETIVO.year} | {SITIO} "
          f"({LAT:.4f}, {LON:.4f}, {ELEV} m) ==========")
    print(final[["pred", "p10", "p90", "verif_POWER", "sin_sesgo_local",
                 "obs_Meteostat"]].round(1))
    print("(p10-p90 = intervalo del 80%; sin_sesgo_local = pronóstico para el Aeropuerto;"
          " obs_Meteostat = lo observado ese día en el Aeropuerto, si ya existe)")
    sens = pd.DataFrame({f"peso {w:.1f}": final["sin_sesgo_local"] + w * final["sesgo_local"]
                         for w in (0.0, 0.5, 1.0)})
    print("\nSensibilidad al peso del sesgo local (PESO_SESGO):")
    print(sens.round(1))

    OUT.mkdir(exist_ok=True)
    final.round(2).to_csv(OUT / f"prediccion_{OBJETIVO.strftime('%Y%m%d')}.csv")
    val.to_csv(OUT / "validacion.csv", index=False)
    graficas(ema, ms_serie, final, val, info_ms["dif"], info_ms["coef"])
    print(f"\nArchivos en {OUT}")


if __name__ == "__main__":
    main()
