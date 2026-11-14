"""
Pipeline Reproducible de Adquisición, Calidad e Integración Multi-Fuente (ELT)
Proyecto: Impacto de las variables meteorológicas en la asistencia presencial y el rendimiento
          en Cálculo Diferencial - Institución Universitaria ITM (2022-1 a 2023-2).
Autores: Camilo Rafael Perez Chaves, Mayra Alejandra Pino Marín
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Rutas relativas según estructura del repositorio ProyectoAdquisicionITM
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
INTEGRATED_DIR = BASE_DIR / "data" / "integrated"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

for folder in [PROCESSED_DIR, INTEGRATED_DIR, FIGURES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def extract_sources():
    """FASE 1 (EXTRACT): Lectura inmutable de las 3 fuentes en data/raw."""
    # Fuente 1: Sensor RFID en aulas/torniquetes (Parquet)
    rfid_path = RAW_DIR / "sensor_rfid_aula_raw.parquet"
    df_rfid_raw = pd.read_parquet(rfid_path)

    # Fuente 2: API Open-Meteo Historical Weather (JSON)
    weather_path = RAW_DIR / "open_meteo_sedes_raw.json"
    with open(weather_path, "r", encoding="utf-8") as f:
        weather_json = json.load(f)

    w_frames = []
    for sede, payload in weather_json["sedes"].items():
        df_sede = pd.DataFrame(payload["hourly"])
        df_sede["campus"] = sede
        df_sede["latitude"] = payload["latitude"]
        df_sede["longitude"] = payload["longitude"]
        w_frames.append(df_sede)
    df_weather_raw = pd.concat(w_frames, ignore_index=True)

    # Fuente 3: Registros Académicos ITM (4 lotes semestrales Excel)
    excel_files = sorted(RAW_DIR.glob("Procesado_*.xlsx"))
    acad_frames = []
    for xfile in excel_files:
        periodo = xfile.stem.replace("Procesado_", "")
        xls = pd.ExcelFile(xfile)
        df_sem = pd.read_excel(xfile, sheet_name=xls.sheet_names[0])
        df_sem["periodo"] = periodo
        acad_frames.append(df_sem)
    df_acad_raw = pd.concat(acad_frames, ignore_index=True)

    return df_rfid_raw, df_weather_raw, df_acad_raw


def transform_and_validate(df_rfid_raw, df_weather_raw, df_acad_raw):
    """FASE 2 (TRANSFORM & DATA QUALITY): Limpieza, reglas de negocio y aserciones."""
    # --- 2.1 Limpieza Sensor RFID ---
    df_rfid = df_rfid_raw.copy()

    # Regla de validez física RSSI UHF: [-90, -20] dBm. Valores +18.5 dBm son picos corruptos.
    rssi_out_bounds = df_rfid["rssi_dbm"].notna() & (
        (df_rfid["rssi_dbm"] < -90.0) | (df_rfid["rssi_dbm"] > -20.0)
    )
    df_rfid.loc[rssi_out_bounds, "rssi_dbm"] = np.nan

    # Deduplicación inteligente por id_evento priorizando registros completos y con RSSI válido
    df_rfid["_q_score"] = (
        df_rfid["estado_asistencia"].notna().astype(int) * 2
        + df_rfid["rssi_dbm"].notna().astype(int)
    )
    df_rfid = (
        df_rfid.sort_values(["id_evento", "_q_score"], ascending=[True, False])
        .drop_duplicates(subset=["id_evento"], keep="first")
        .drop(columns=["_q_score"])
        .reset_index(drop=True)
    )

    # Cálculo de retraso en minutos respecto al inicio del bloque horario
    sched_dt = pd.to_datetime(df_rfid["fecha"] + " " + df_rfid["bloque_hora"])
    delay_min = (df_rfid["timestamp_entrada"] - sched_dt).dt.total_seconds() / 60.0
    df_rfid["retraso_minutos"] = np.where(
        df_rfid["timestamp_entrada"].notna(), delay_min, np.nan
    )

    # Imputación determinística de estado_asistencia nulo (regla umbral 15 min)
    null_est = df_rfid["estado_asistencia"].isna()
    df_rfid.loc[null_est & df_rfid["timestamp_entrada"].isna(), "estado_asistencia"] = "AUSENTE"
    df_rfid.loc[null_est & (df_rfid["retraso_minutos"] <= 15.0), "estado_asistencia"] = "PRESENTE"
    df_rfid.loc[null_est & (df_rfid["retraso_minutos"] > 15.0), "estado_asistencia"] = "TARDE"

    # Imputación de RSSI corrupto/nulo en asistentes mediante mediana por campus
    present_mask = df_rfid["estado_asistencia"].isin(["PRESENTE", "TARDE"])
    df_rfid["rssi_dbm"] = df_rfid.groupby("campus")["rssi_dbm"].transform(
        lambda s: s.fillna(s.median())
    )
    df_rfid.loc[~present_mask, "rssi_dbm"] = np.nan
    df_rfid["fecha_hora_bloque"] = df_rfid["fecha"] + "T" + df_rfid["bloque_hora"]

    # Aserciones de calidad RFID
    assert df_rfid["id_evento"].duplicated().sum() == 0, "Error: id_evento duplicados"
    assert df_rfid["estado_asistencia"].isna().sum() == 0, "Error: estado_asistencia nulo"
    assert df_rfid.loc[present_mask, "rssi_dbm"].between(-90, -20).all(), "Error: RSSI fuera de rango"

    # --- 2.2 Limpieza API Open-Meteo ---
    df_weather = df_weather_raw.copy()
    df_weather["time_dt"] = pd.to_datetime(df_weather["time"])
    df_weather["fecha"] = df_weather["time_dt"].dt.strftime("%Y-%m-%d")
    df_weather = df_weather.sort_values(["campus", "time_dt"]).reset_index(drop=True)

    # Ventana móvil de precipitación acumulada en las últimas 3 horas (afecta desplazamiento al campus)
    df_weather["precip_3h_acum_mm"] = (
        df_weather.groupby("campus")["precipitation"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).sum())
    )
    df_daily = df_weather.groupby(["campus", "fecha"], as_index=False).agg(
        temp_prom_diaria_c=("temperature_2m", "mean"),
        precip_total_diaria_mm=("precipitation", "sum"),
        precip_max_horaria_mm=("precipitation", "max"),
    )
    df_weather = df_weather.merge(df_daily, on=["campus", "fecha"], how="left", validate="m:1")

    # Aserciones de calidad Clima
    assert df_weather.duplicated(subset=["campus", "time"]).sum() == 0, "Error: Clima duplicado"
    assert df_weather["temperature_2m"].between(5.0, 40.0).all(), "Error: Temperatura fuera de rango"
    assert (df_weather["precipitation"] >= 0.0).all(), "Error: Precipitación negativa"

    # --- 2.3 Limpieza Datos Académicos ---
    df_acad = df_acad_raw.copy()
    df_acad["campus_norm"] = df_acad["Campus"].replace({"FRATERNIDAD MEDELLÍN": "FRATERNIDAD"})

    nota_cols = ["Nota 1", "Nota 2", "Nota 3", "Nota 4", "Nota 5", "Nota 6"]
    all_zero = df_acad[nota_cols].apply(lambda c: c.astype(str).str.endswith("-0")).all(axis=1)
    null_res = df_acad["Resultado"].isna()

    df_acad.loc[null_res & all_zero & (df_acad["Final"] == 0.0), "Resultado"] = "Canceló"
    df_acad.loc[df_acad["Resultado"].isna() & (df_acad["Final"] < 3.0), "Resultado"] = "Reprobó"
    df_acad.loc[df_acad["Resultado"].isna() & (df_acad["Final"] >= 3.0), "Resultado"] = "Aprobó"

    df_acad["aprobo"] = df_acad["Resultado"].isin(["Aprobó", "Habilitó y Aprobó"]).astype(int)
    df_acad["deserto_o_cancelo"] = df_acad["Resultado"].isin(["Canceló", "Reprobó Inasistencia"]).astype(int)

    # Aserciones de calidad Académica
    assert df_acad.duplicated(subset=["periodo", "ID_Anonimizado"]).sum() == 0, "Error: Matrícula duplicada"
    assert df_acad["Final"].between(0.0, 5.0).all(), "Error: Nota Final fuera de [0, 5]"
    assert df_acad["Resultado"].isna().sum() == 0, "Error: Resultado nulo"

    # Guardar capa Silver (data/processed)
    df_rfid.to_parquet(PROCESSED_DIR / "rfid_clean.parquet", index=False)
    df_weather.to_parquet(PROCESSED_DIR / "weather_clean.parquet", index=False)
    df_acad.to_parquet(PROCESSED_DIR / "academic_clean.parquet", index=False)

    return df_rfid, df_weather, df_acad


def integrate_and_enrich(df_rfid, df_weather, df_acad):
    """FASE 3 (INTEGRATE & KPIs): Unión multi-fuente controlada y generación de capa Gold."""
    # Paso 1: Sensor RFID (68,918) LEFT JOIN Open-Meteo (70,080) por (campus, fecha_hora_bloque)
    df_step1 = pd.merge(
        df_rfid,
        df_weather.drop(columns=["fecha", "time_dt"]),
        left_on=["campus", "fecha_hora_bloque"],
        right_on=["campus", "time"],
        how="left",
        validate="m:1",
        indicator="_merge_clima",
    )
    assert (df_step1["_merge_clima"] == "both").all(), "Error: Eventos RFID sin cruce climático"
    df_step1 = df_step1.drop(columns=["_merge_clima", "time"])

    # Paso 2: Resultado Paso 1 (68,918) LEFT JOIN Académico (2,045) por (periodo, id_estudiante)
    acad_subset = df_acad[
        ["periodo", "ID_Anonimizado", "campus_norm", "Sexo", "Final", "Resultado", "aprobo", "deserto_o_cancelo"]
    ]
    df_gold = pd.merge(
        df_step1,
        acad_subset,
        left_on=["periodo", "id_estudiante"],
        right_on=["periodo", "ID_Anonimizado"],
        how="left",
        validate="m:1",
        indicator="_merge_acad",
    )
    assert (df_gold["_merge_acad"] == "both").all(), "Error: Eventos RFID sin matrícula académica"
    df_gold = df_gold.drop(columns=["_merge_acad", "ID_Anonimizado"])

    # KPIs a nivel de evento
    df_gold["asistio_bin"] = df_gold["estado_asistencia"].isin(["PRESENTE", "TARDE"]).astype(int)
    df_gold["tarde_bin"] = (df_gold["estado_asistencia"] == "TARDE").astype(int)
    df_gold["condicion_lluvia"] = pd.cut(
        df_gold["precip_3h_acum_mm"],
        bins=[-0.01, 0.0, 2.5, 100.0],
        labels=["Sin lluvia (0 mm)", "Lluvia ligera (0.1–2.5 mm)", "Lluvia intensa (>2.5 mm)"],
    )

    # Agregación analítica a nivel estudiante-semestre (2,027 matrículas en campus ITM)
    df_student_sem = df_gold.groupby(["periodo", "id_estudiante", "campus", "Sexo"], as_index=False).agg(
        sesiones_programadas=("id_evento", "count"),
        sesiones_asistidas=("asistio_bin", "sum"),
        sesiones_tarde=("tarde_bin", "sum"),
        precip_prom_clase_mm=("precip_3h_acum_mm", "mean"),
        sesiones_lluvia_intensa=("precip_3h_acum_mm", lambda s: (s > 2.5).sum()),
        Final=("Final", "first"),
        Resultado=("Resultado", "first"),
        aprobo=("aprobo", "first"),
        deserto_o_cancelo=("deserto_o_cancelo", "first"),
    )
    df_student_sem["tasa_asistencia_pct"] = (
        df_student_sem["sesiones_asistidas"] / df_student_sem["sesiones_programadas"] * 100.0
    )
    df_student_sem["tasa_puntualidad_pct"] = np.where(
        df_student_sem["sesiones_asistidas"] > 0,
        (df_student_sem["sesiones_asistidas"] - df_student_sem["sesiones_tarde"])
        / df_student_sem["sesiones_asistidas"]
        * 100.0,
        0.0,
    )

    # Guardar capa Gold (data/integrated)
    df_gold.to_parquet(INTEGRATED_DIR / "itm_calculo_eventos_integrado.parquet", index=False)
    df_gold.to_csv(INTEGRATED_DIR / "itm_calculo_eventos_integrado.csv", index=False)
    df_student_sem.to_parquet(INTEGRATED_DIR / "itm_calculo_estudiantes_kpi.parquet", index=False)
    df_student_sem.to_csv(INTEGRATED_DIR / "itm_calculo_estudiantes_kpi.csv", index=False)

    return df_gold, df_student_sem


def generate_visualizations(df_gold, df_student_sem):
    """FASE 4 (VISUALIZATION): Genera las 3 figuras analíticas en outputs/figures."""
    w_sum = df_gold.groupby("condicion_lluvia", observed=False).agg(
        n=("id_evento", "count"),
        att=("asistio_bin", lambda s: s.mean() * 100),
        abs_r=("asistio_bin", lambda s: (1 - s.mean()) * 100),
        tard=("tarde_bin", lambda s: s.sum() / max(1, (df_gold.loc[s.index, "asistio_bin"] == 1).sum()) * 100),
        delay=("retraso_minutos", "mean"),
    )

    # Figura 1: Impacto de la precipitación en asistencia y tardanza
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=220)
    cats = ["Sin lluvia\n(0 mm)", "Lluvia ligera\n(0.1–2.5 mm)", "Lluvia intensa\n(>2.5 mm)"]
    x = np.arange(len(cats))
    w = 0.36
    b1 = ax1.bar(x - w / 2, w_sum["att"], width=w, color="#1B4F72", label="Tasa de Asistencia (%)")
    b2 = ax1.bar(x + w / 2, w_sum["abs_r"], width=w, color="#C0392B", label="Tasa de Ausentismo (%)")
    ax1.set_title("A. Asistencia vs. Ausentismo según Precipitación (3h)", fontsize=11.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}\n(n={n:,})" for c, n in zip(cats, w_sum["n"])])
    ax1.set_ylabel("Porcentaje de Sesiones (%)")
    ax1.set_ylim(0, 90)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(frameon=False)
    for bar in list(b1) + list(b2):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")

    b3 = ax2.bar(x, w_sum["tard"], width=0.48, color=["#2E86C1", "#F39C12", "#D35400"])
    ax2.set_title("B. Tasa de Tardanza (>15 min) y Retraso Promedio", fontsize=11.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats)
    ax2.set_ylabel("Tardanzas sobre Asistentes (%)")
    ax2.set_ylim(0, 28)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, dly in zip(b3, w_sum["delay"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7, f"{bar.get_height():.2f}%\n({dly:.1f} min)", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_clima_asistencia_tardanza.png", dpi=220)
    plt.close()

    # Figura 2: Desenlace académico por rango de asistencia y correlación
    df_student_sem["rango_asistencia"] = pd.cut(
        df_student_sem["tasa_asistencia_pct"],
        bins=[-0.1, 50, 75, 85, 100.1],
        labels=["Crítica\n(≤50%)", "Baja\n(50–75%)", "Media\n(75–85%)", "Alta\n(>85%)"],
    )
    b_sum = df_student_sem.groupby("rango_asistencia", observed=False).agg(
        n=("id_estudiante", "count"),
        aprob=("aprobo", lambda s: s.mean() * 100),
        cancel=("deserto_o_cancelo", lambda s: s.mean() * 100),
    )
    r_val, _ = stats.pearsonr(df_student_sem["tasa_asistencia_pct"], df_student_sem["Final"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=220)
    x2 = np.arange(len(b_sum))
    b1 = ax1.bar(x2 - w / 2, b_sum["aprob"], width=w, color="#1E8449", label="Tasa de Aprobación (%)")
    b2 = ax1.bar(x2 + w / 2, b_sum["cancel"], width=w, color="#922B21", label="Cancelación / Reprob. Inasist. (%)")
    ax1.set_title("A. Desenlace Académico por Rango de Asistencia", fontsize=11.5, fontweight="bold")
    ax1.set_xticks(x2)
    ax1.set_xticklabels([f"{idx}\n(n={n})" for idx, n in zip(b_sum.index, b_sum["n"])])
    ax1.set_ylabel("Porcentaje de Estudiantes (%)")
    ax1.set_ylim(0, 112)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(frameon=False, loc="upper center")
    for bar in list(b1) + list(b2):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center", fontsize=8.8, fontweight="bold")

    colors_pts = np.where(df_student_sem["aprobo"] == 1, "#1E8449", "#C0392B")
    ax2.scatter(df_student_sem["tasa_asistencia_pct"], df_student_sem["Final"], c=colors_pts, alpha=0.35, s=18)
    m, b_int = np.polyfit(df_student_sem["tasa_asistencia_pct"], df_student_sem["Final"], 1)
    xl = np.linspace(df_student_sem["tasa_asistencia_pct"].min(), 100, 100)
    ax2.plot(xl, m * xl + b_int, color="#1B4F72", linewidth=2.2, label=f"Regresión lineal (r = {r_val:.3f})")
    ax2.axhline(3.0, color="#555555", linestyle=":", linewidth=1.2, label="Nota mínima aprobatoria (3.0)")
    ax2.set_title("B. Correlación Asistencia Semestral vs. Nota Final", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Tasa de Asistencia Presencial Semestral (%)")
    ax2.set_ylabel("Nota Definitiva (0.0 – 5.0)")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_asistencia_rendimiento_calculo.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    rfid_r, weather_r, acad_r = extract_sources()
    rfid_c, weather_c, acad_c = transform_and_validate(rfid_r, weather_r, acad_r)
    gold_events, gold_students = integrate_and_enrich(rfid_c, weather_c, acad_c)
    generate_visualizations(gold_events, gold_students)
    print(f"Pipeline ejecutado con éxito. Eventos integrados: {len(gold_events):,} | Estudiantes-semestre: {len(gold_students):,}")