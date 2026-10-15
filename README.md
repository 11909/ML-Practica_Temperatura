# ML-Practica_Temperatura
Práctica de predicción de temperatura.

**Objetivo:** pronosticar la temperatura del aire el **27 de octubre de 2026** a las **07:00, 07:30, 08:00 y 08:30**
en el punto 19.505484, -99.146543 (2240 m), Zacatenco, CDMX.

## Resultado (corrida del 2026-10-14)

| Hora  | Pronóstico | Intervalo 80 % | Verificación NASA POWER |
|-------|-----------:|:--------------:|------------------------:|
| 07:00 | **16.8 °C** | 13.8 – 19.8 | 16.2 °C |
| 07:30 | **17.4 °C** | 14.4 – 20.4 | 16.4 °C |
| 08:00 | **18.2 °C** | 15.2 – 21.2 | 17.0 °C |
| 08:30 | **19.1 °C** | 16.1 – 22.1 | 18.0 °C |

Los valores son "como los mediría la EMA ENCB II", porque el modelo se corrige con esa estación.

## Cómo ejecutar
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py
```
La primera vez descarga los datos (se guardan en `data/`). Los resultados quedan en `outputs/`:
`prediccion_27oct.csv`, `validacion.csv` y `resumen.png`.

## Por qué este enfoque

**1. El problema es climatológico, no de pronóstico del tiempo.** Faltan 13 días para el 27 de octubre, y a ese
plazo los modelos meteorológicos ya casi no tienen habilidad. Lo mejor que se puede hacer es estimar cómo es una
mañana típica de finales de octubre en el sitio y ajustarla con la información reciente.

**2. La EMA sola no alcanza.** `Dataset_EMA_ENCB.csv` (ENCB II, 19.49845, -99.14489, 2236 m, a ~0.8 km del punto
objetivo) sólo cubre del 19 de julio al 12 de agosto de 2026: temporada de lluvias y apenas ~16 días completos.
Con eso no se puede aprender cómo es octubre. Por eso:

- **Meteostat** (estación Aeropuerto, 10 km, 2234 m, con huecos rellenados con Tacubaya) aporta datos horarios
  completos desde 2017 hasta hoy. Antes de 2017 sólo hay ~4 lecturas por día, así que esos años no se usan.
- **NASA POWER** (reanálisis MERRA-2, celda de ~50 km) aporta 2005–2026. Es una malla gruesa y su altitud es de
  ~2600 m, por lo que sale más fría; se usa como verificación independiente.
- **La EMA** se usa para medir cuánto difiere el sitio de esas fuentes (el "sesgo local").

**3. El modelo** (todo en `main.py`):

```
pronóstico = climatología(fecha, hora) + alfa · anomalía reciente + sesgo local
```

- *Climatología*: regresión Ridge sobre la serie horaria de Meteostat. Las variables son senos y cosenos del día
  del año y de la hora (así el 31-dic queda junto al 1-ene y las 23:00 junto a las 00:00), sus productos (el ciclo
  diario cambia con la estación) y una tendencia anual. Como la hora es continua, las 07:30 y 08:30 se predicen
  directamente. Se comparó con Gradient Boosting y Ridge fue mejor en validación (MAE 1.76 vs 2.03 °C).
- *Anomalía reciente*: cuánto más caliente o frío que lo normal estuvieron las mañanas de los últimos 21 días.
  `alfa` se aprende con regresión sobre los años anteriores: es cuánto de esa anomalía sigue presente 13 días
  después. Salió `alfa` = 0.11: a dos semanas la atmósfera "recuerda" muy poco.
- *Sesgo local*: EMA − Meteostat, ajustado en función de la **altura del sol** y no de la hora del reloj. En julio,
  a las 07:00 el sol ya lleva ~50 min fuera; el 27 de octubre a esa hora apenas está saliendo (altura ~5°). Así se
  compara el mismo momento del calentamiento matutino. Resultó de unos **+3.1 °C** en las horas objetivo.

**4. Validación** (para saber cuánto confiar):

- *Dejando un año fuera, 2018–2025*: se entrena sin ese año y se pronostican las mañanas del 20-oct al 3-nov,
  emitiendo 13 días antes. MAE del modelo = **1.76 °C**, contra 1.79 °C de una climatología simple y 1.96 °C de
  la persistencia (repetir la última semana). El modelo apenas mejora a la climatología. Es lo esperable a
  13 días de plazo.
- *Sesgo local en días no vistos de la EMA* (ajuste con 19–31 jul, prueba con 1–12 ago): sin corrección, el error
  medio es de −2.7 °C; con corrección, el MAE baja a **1.1 °C**.
- El intervalo del 80 % combina el error de validación y la variabilidad diaria del sesgo (±3 °C).

## Limitaciones
- **El sesgo local se midió en julio–agosto** (lluvias) y se aplica a octubre (inicio de secas). Si cambia con
  la estación, el pronóstico se desplaza. En la prueba de agosto el modelo quedó +0.7 °C caliente, señal de que
  ese sesgo puede variar.
- **La EMA parece leer alto:** máximas de 33–35 °C en temporada de lluvias, ~6 °C más que el Aeropuerto en la
  tarde. Puede deberse a la protección del sensor o a la ubicación. El pronóstico reproduce lo que marcaría la EMA,
  no necesariamente la temperatura "real" del aire.
- **Un frente frío ("norte")** puede bajar 3–5 °C las mañanas. A 13 días eso no se puede prever, y por eso el
  intervalo es amplio.
- **Para mejorar:** volver a correr el modelo cerca del 27-oct (la anomalía reciente se actualiza sola) y, sobre
  todo, conseguir datos de la EMA de septiembre–octubre para medir el sesgo en la temporada correcta.
