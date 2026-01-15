import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from PIL import Image
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google import genai
import time




# Cargar variables de entorno
load_dotenv()

# Configuración inicial de la página
st.set_page_config(page_title="ATP - Dashboard TI", layout="wide")

def traducir_error(codigo, razon=None):
    if codigo == "200 TODO OK":
        return "200 TODO OK"
    descripcion = codigo
    # Para códigos específicos, usar descripción fija para unificar
    if codigo == "200-63":  # Reserva no activa
        return "200-63: Reserva no activa"
    if codigo == "200-68":  # ONT ya existe en el inventario
        return "200-68: La ONT ya existe en el inventario"
    if codigo == "200-39":  # ONT ya asignada
        return "200-39: La reserva ya tiene una ONT asignada"
    if codigo == "100-5":  # Timeout
        return "100-5: REQUEST_TIME_OUT"
    if razon and razon.strip() and razon.lower() not in ['nan', 'none', '']:
        razon_limpia = razon.strip()
        # Para 200-46, truncar la razón hasta el comando
        if codigo == "200-46" and "debido a" in razon_limpia:
            razon_limpia = razon_limpia.split("debido a")[0].strip()
        descripcion = f"{codigo}: {razon_limpia}"
    return descripcion

def limpiar_encabezado(df):
    nuevos_nombres = {col: re.sub(r'[^a-zA-Z0-9_]', '', col).lower() for col in df.columns}
    return df.rename(columns=nuevos_nombres)


def generar_comentario(api_key, errores):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=api_key)

            prompt = (
                
                "Actúa como Analista Senior de Telecomunicaciones. Genera un comentario técnico ultra-breve.\n\n"
                "REGLAS:\n"
                "1. Ignora '200 TODO OK'. Solo analiza los errores.\n"
                "2. Identifica las 2 mayores causas. Suma todas las demás en 'Otros'.\n"
                "3. La suma de los porcentajes DEBE ser igual al % total de No Exitosas.\n"
                "4. Formato estricto: 'Causal1 (X%), Causal2 (Y%) y Otros (Z%)'.\n"
                "5. Si no hay errores, responde: 'Sin errores registrados'.\n"
                "6. Abrevia términos (Ej: 'ONT en inv.', 'Timeout', 'Op. en proc.').\n"
                "7. Máximo 12 palabras.\n\n"
                "DATOS DEL PROCESO:\n"
    f"{errores}"
    )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            return response.text.strip()

        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "UNAVAILABLE" in error_msg or "overloaded" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Backoff: 2s, 4s, 6s
                    time.sleep(wait_time)
                    continue
            return f"Error de IA: {error_msg}"

# --- 1. ENCABEZADO CON LOGO ---
ruta_logo = r"C:\Users\FelipeCruz\OneDrive - Andean Telecom Partners\Escritorio\Reportes_ATP\logo_ATP.png"

header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    try:
        img = Image.open(ruta_logo)
        st.image(img, width=180) 
    except:
        st.write("⚠️ Logo ATP")

with header_col2:
    st.markdown("<h1 style='margin-bottom: 0;'>📊 ESTADO TRANSACCIONES SISTEMAS TI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: gray; margin-top: 0;'>OPERACIÓN COMPARTIDA CLARO</h3>", unsafe_allow_html=True)

# Lógica de API Key en Sidebar
st.sidebar.title("Configuración")
api_token = os.getenv("GEMINI_API_KEY")

if not api_token:
    api_token = st.sidebar.text_input("Introduce tu Gemini API Key:", type="password")
    if api_token:
        st.sidebar.success("Clave API cargada manualmente")

uploaded_file = st.sidebar.file_uploader("📂 Cargar CSV de Kibana", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, encoding='latin-1')
    df_clean = limpiar_encabezado(df_raw)
    
    MAPEO = {
        'addont': 'Reservas', 'addservice': 'Activación',
        'changeont': 'Cambio ONT', 'deleteont': 'Terminación',
        'neighborsquery': 'Soporte N1 (Consulta Vecinos)', '-': 'Error de solicitud'
    }

    COL_TRANS, COL_CODE, COL_TIME, COL_REASON = 'requestdescription', 'responsecode', 'timestamp', 'responsereason'
    
    df_clean[COL_TRANS] = df_clean[COL_TRANS].astype(str).str.strip().str.lower().replace(MAPEO)
    df_clean[COL_CODE] = df_clean[COL_CODE].astype(str).str.strip().replace(['nan', 'NaN', 'None', ''], '-')
    df_clean['Es_Exito'] = df_clean[COL_CODE].isin(['.', '-'])
    df_clean[COL_CODE] = df_clean[COL_CODE].apply(lambda x: "200 TODO OK" if x in ['.', '-'] else x)
    
    df_clean[COL_CODE] = df_clean.apply(lambda row: traducir_error(row[COL_CODE], row.get(COL_REASON)), axis=1)

    if COL_TIME in df_clean.columns:
        df_clean[COL_TIME] = pd.to_datetime(df_clean[COL_TIME].astype(str).str.replace(' @ ', ' '), errors='coerce')
        df_clean = df_clean.dropna(subset=[COL_TIME])
        rango_input = st.sidebar.date_input("Filtrar Periodo", [df_clean[COL_TIME].min().date(), df_clean[COL_TIME].max().date()])
        
        if len(rango_input) == 2:
            df_filtrado = df_clean[(df_clean[COL_TIME].dt.date >= rango_input[0]) & (df_clean[COL_TIME].dt.date <= rango_input[1])].copy()
            periodo_str = f"{rango_input[0]} al {rango_input[1]}"
        else:
            df_filtrado = df_clean.copy()
            periodo_str = "Seleccione un rango"
    else:
        df_filtrado = df_clean.copy()
        periodo_str = "N/A"

    # --- 2. KPIs ---
    resumen = df_filtrado.groupby(COL_TRANS).agg(Total=('Es_Exito', 'size'), Exitosas=('Es_Exito', 'sum')).reset_index()
    resumen['No Exitosas'] = resumen['Total'] - resumen['Exitosas']
    resumen['% Éxito'] = ((resumen['Exitosas'] / resumen['Total']) * 100).round(1)
    resumen['% No Exitosas'] = ((resumen['No Exitosas'] / resumen['Total']) * 100).round(1)

    t_gen = int(resumen['Total'].sum())
    e_gen = (resumen['Exitosas'].sum() / t_gen * 100) if t_gen > 0 else 0

    st.markdown("""<style>[data-testid="stMetricValue"] { font-size: 26px !important; }</style>""", unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    k1.metric("TOTAL TRANSACCIONES", f"{t_gen:,}")
    k2.metric("SLA GLOBAL ÉXITO", f"{e_gen:.2f}%")
    with k3:
        st.write("**PERIODO ANALIZADO**")
        st.caption(periodo_str)

    # --- 3. GRÁFICA ---
    st.write("---")
    fig, ax1 = plt.subplots(figsize=(14, 7))
    bars = ax1.bar(resumen[COL_TRANS], resumen['Total'], color='#A9A9A9', alpha=0.7, label='Q Transacciones')
    ax1.set_ylabel('Volumen de Transacciones', fontweight='bold', color='#4F4F4F', fontsize=12)
    
    ax2 = ax1.twinx()
    line = ax2.plot(resumen[COL_TRANS], resumen['% Éxito'], color='#C00000', marker='o', linewidth=3, markersize=10, label='% Éxito (SLA)')
    ax2.set_ylabel('% Éxito (SLA)', color='#C00000', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 120) 

    max_y = resumen['Total'].max()
    for i, bar in enumerate(bars):
        proc = resumen[COL_TRANS][i]
        height = bar.get_height()
        if proc.lower() in ['activación', 'terminación']:
            # En el medio
            ax1.text(bar.get_x() + bar.get_width()/2., height / 2, f'{int(height)}', ha='center', va='center', fontweight='bold', color='black')
        else:
            # Arriba de la barra
            ax1.text(bar.get_x() + bar.get_width()/2., height + (max_y * 0.01), f'{int(height)}', ha='center', va='bottom', fontweight='bold', color='black')

    for i, val in enumerate(resumen['% Éxito']):
        ax2.annotate(f"{val}%", (resumen[COL_TRANS][i], resumen['% Éxito'][i]), textcoords="offset points", xytext=(0, 15), ha='center', color='#C00000', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C00000", alpha=0.9))

    plt.title("ANÁLISIS DE VOLUMETRÍA Y TASA DE ÉXITO", fontweight='bold', fontsize=16, pad=20)
    st.pyplot(fig)

    # --- 4. CONSOLIDADO CON IA ---
    st.write("---")
    st.markdown("### 📝 Consolidado de Operación")
    
    current_key = f"{uploaded_file.name}_{rango_input[0]}_{rango_input[1]}" if len(rango_input) == 2 else f"{uploaded_file.name}_all"
    
    if 'resumen_df' not in st.session_state or st.session_state.get('last_key') != current_key:
        if api_token:
            with st.spinner("Generando comentarios con IA..."):
                for idx in resumen.index:
                    proc = resumen.loc[idx, COL_TRANS]
                    df_proc = df_filtrado[df_filtrado[COL_TRANS] == proc]
                    df_err = df_proc[df_proc[COL_CODE] != "200 TODO OK"]
                    if not df_err.empty:
                        errores_df = df_err[COL_CODE].value_counts().reset_index()
                        errores_df.columns = ['Código', 'Cantidad']
                        errores_df['%'] = (errores_df['Cantidad'] / len(df_proc) * 100).round(1)
                        errores_str = f"Proceso: {proc}\n" + "\n".join([f"{row['Código']}: {row['Cantidad']} transacciones ({row['%']}%)" for _, row in errores_df.iterrows()])
                        comentario = generar_comentario(api_token, errores_str)
                        resumen.at[idx, 'Causal No Exitosa (Comentario)'] = comentario
                    else:
                        resumen.at[idx, 'Causal No Exitosa (Comentario)'] = "Sin errores registrados"
        else:
            resumen['Causal No Exitosa (Comentario)'] = ""
        st.session_state['resumen_df'] = resumen
        st.session_state['last_key'] = current_key

    columnas_mostrar = [COL_TRANS, 'Total', 'Exitosas', 'No Exitosas', '% Éxito', '% No Exitosas', 'Causal No Exitosa (Comentario)']
    
    # Ensure the column exists (in case of old session state)
    if 'Causal No Exitosa (Comentario)' not in st.session_state['resumen_df'].columns:
        st.session_state['resumen_df']['Causal No Exitosa (Comentario)'] = ""
    
    df_editor = st.data_editor(
        st.session_state['resumen_df'][columnas_mostrar],
        column_config={
            "% Éxito": st.column_config.NumberColumn(format="%.1f%%"),
            "% No Exitosas": st.column_config.NumberColumn(format="%.1f%%"),
            "Causal No Exitosa (Comentario)": st.column_config.TextColumn("Análisis IA / Comentarios", width="large")
        },
        hide_index=True, width='stretch'
    )
    st.session_state['resumen_df'] = df_editor

    st.write("---")
    st.markdown("### 🛠️ Comportamiento de Errores por Proceso")
    
    cols_grid = st.columns(2)
    for i, proc in enumerate(resumen[COL_TRANS].unique()):
        with cols_grid[i % 2]:
            st.markdown(f"**Análisis: {proc}**")
            df_proc = df_filtrado[df_filtrado[COL_TRANS] == proc]
            detalle = df_proc[COL_CODE].value_counts().reset_index()
            detalle.columns = ['Descripción / Código', 'Cantidad']
            detalle['%'] = (detalle['Cantidad'] / len(df_proc) * 100).map("{:.1f}%".format)
            st.table(detalle)
    
    
    # --- 5. DETALLES ---
    st.write("---")
    st.markdown("### 🚩 Top 5 Errores Críticos")
    df_solo_errores = df_filtrado[df_filtrado[COL_CODE] != "200 TODO OK"]
    if not df_solo_errores.empty:
        top_errores = df_solo_errores[COL_CODE].value_counts().head(5)
        
        # Gráfica de torta
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        ax_pie.pie(top_errores.values, autopct='%1.1f%%', pctdistance=0.85, startangle=70, colors=plt.cm.Paired.colors, textprops={'fontsize': 9})
        ax_pie.set_title("Distribución de Top 5 Errores Críticos", fontsize=14)
        ax_pie.axis('equal')  # Para que sea un círculo
        
        # Mostrar gráfica y tabla lado a lado
        col_pie, col_table = st.columns([2, 1])  # Gráfica más ancha, tabla más estrecha
        with col_pie:
            st.pyplot(fig_pie, width=600)
        with col_table:
            st.markdown("**Leyenda de Colores:**")
            colors = plt.cm.Paired.colors[:len(top_errores.index)]
            for i, error in enumerate(top_errores.index):
                color = colors[i]
                color_hex = '#%02x%02x%02x' % (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                st.markdown(f"<span style='color:{color_hex}; font-size:20px;'>■</span> {error}", unsafe_allow_html=True)
            st.markdown("**Detalle Numérico:**")
            top_errores_df = top_errores.reset_index()
            top_errores_df.columns = ['Error', 'Frecuencia']
            st.table(top_errores_df)

    def generar_resumen_ejecutivo(api_key, resumen_datos):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = genai.Client(api_key=api_key)

                prompt = (
                    "Actúa como Analista Senior de Telecomunicaciones. Genera un RESUMEN EJECUTIVO directo.\n\n"
                    "ESTRUCTURA Y REGLAS:\n"
                    "1. Usa una lista de máximo 4 puntos (bullets).\n"
                    "2. El primer punto DEBE mencionar el volumen total y el % de éxito general.\n"
                    "3. El segundo punto DEBE identificar el proceso más crítico (Cambio ONT o Reservas) y su causa principal.\n"
                    "4. El tercer punto debe mencionar cualquier hallazgo técnico relevante (como errores en comandos OLT).\n"
                    "5. PROHIBIDO: Introducciones, saludos o frases como 'Aquí tienes el resumen'.\n"
                    "6. Lenguaje: Técnico-ejecutivo (ej. 'Fricción en inventario', 'Degradación de éxito', 'Sincronización OLT').\n\n"
                    "DATOS CONSOLIDADOS:\n"
                    f"{resumen_datos}"
)

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                return response.text.strip()

            except Exception as e:
                return f"No fue posible generar el resumen ejecutivo: {e}"
    # === Datos consolidados para Resumen Ejecutivo ===

    total_tx = df_filtrado.shape[0]
    pct_no_exitosas = 100 - e_gen

    proceso_mas_fallas = (
        resumen.sort_values('No Exitosas', ascending=False)
        .iloc[0][COL_TRANS]
        if not resumen.empty else "N/A"
    )

    top_causas = (
        df_solo_errores[COL_CODE]
        .value_counts()
        .head(3)
        .index
        .tolist()
        if not df_solo_errores.empty else []
    )

    resumen_datos = f"""
    Total de transacciones: {total_tx}
    SLA global de éxito: {e_gen:.1f}%
    Porcentaje no exitosas: {pct_no_exitosas:.1f}%
    Proceso con mayor impacto: {proceso_mas_fallas}
    Principales causas: {', '.join(top_causas)}
    """

    st.write("---")
    st.markdown("### 📌 Resumen Ejecutivo del Período (Generado con IA)")

    if api_token:
        with st.spinner("Generando análisis ejecutivo..."):
            resumen_ejecutivo = generar_resumen_ejecutivo(
                api_key=api_token,
                resumen_datos=resumen_datos
            )

        for linea in resumen_ejecutivo.split("\n"):
            if linea.strip():
                st.markdown(f"- {linea.strip()}")
    else:
        st.info("Ingrese una API Key para generar el resumen ejecutivo.")

else:
    st.info("👋 Por favor, carga el archivo CSV para comenzar.")