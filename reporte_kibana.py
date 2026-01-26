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
import io
import base64

def fig_to_base64(fig):
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    return base64.b64encode(img_buf.read()).decode('utf-8')


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
    if codigo == "200-15":  # Timeout
        return "200-15: No existe la reserva para el VNO indicado"
    if codigo == "200-38":  
        return "200-38: El servicio no está asociado al VNO"
    if codigo == "100-191":  
        return "100-191: No se asociaron servicios a ONT nueva; rollback ejecutado"
    if codigo == "100-190":  
        return "100-190: No se creo la ONT nueva; rollback ejecutado"
    if codigo == "200-41":
        return "200-41: Servicio sin componente asociado"
    if codigo == "100-2":
        return "100-2: Falta parámetro requerido"



    if razon and razon.strip() and razon.lower() not in ['nan', 'none', '']:
        razon_limpia = razon.strip()
        # Para 200-46, truncar la razón hasta el comando
        if codigo == "200-46" and "debido a" in razon_limpia:
            razon_limpia = razon_limpia.split("debido a")[0].strip()
        descripcion = f"{codigo}: {razon_limpia}"
        # Para 200-40, truncar la razón hasta el comando
        if codigo == "200-40" and "no pueden" in razon_limpia:
            razon_limpia = razon_limpia.split("no pueden")[0].strip()
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


def generar_resumen_ejecutivo(api_key, resumen_datos):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=api_key)

            prompt = (
                "Actúa como Director de Operaciones de Telecomunicaciones. "
                "Genera un resumen ejecutivo de 3-5 líneas (bullets) basado en los datos proporcionados.\n\n"
                "REGLAS:\n"
                "1. Sé conciso y actionable.\n"
                "2. Destaca los puntos críticos: SLA, proceso problemático, causas principales.\n"
                "3. Usa lenguaje ejecutivo (sin detalles técnicos innecesarios).\n"
                "4. Cada línea debe ser independiente (formato bullet).\n\n"
                "DATOS DEL PERÍODO:\n"
                f"{resumen_datos}"
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



OPERADOR_LABEL = {
    "Ambos": "CLARO / ETB",
    "CLARO": "CLARO",
    "ETB": "ETB"
}

with header_col2:
    st.markdown("<h1 style='margin-bottom: 0; padding-left: 100px;'>📊 ESTADO TRANSACCIONES SISTEMAS TI</h1>", unsafe_allow_html=True)

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
    
    # Agregar filtro de operador
    operador_filtro = st.sidebar.selectbox(
        "🏢 Filtrar por Operador",
        ["Ambos", "CLARO", "ETB"]
    )
    with header_col2:
        
        st.markdown(f"<h3 style='color: gray; margin-top: 0px; padding-left: 200px;'>OPERACIÓN COMPARTIDA {OPERADOR_LABEL[operador_filtro]}</h3>", unsafe_allow_html=True)

    MAPEO = {
        'addont': 'Reservas', 'addservice': 'Activación',
        'changeont': 'Cambio ONT', 'deleteont': 'Terminación',
        'neighborsquery': 'Soporte N1 Vecinos', '-': 'Error de solicitud', 'activateservice': 'Reconexión',
        'bandwidthquery': 'Ancho de banda', 'signallevelquery': 'Consulta niveles', 'statusquery': 'Estado servicio',
        'suspendservice': 'Suspensión ', 'deleteservice': 'Eliminación servicio'
    }

    COL_TRANS, COL_CODE, COL_TIME, COL_REASON = 'requestdescription', 'responsecode', 'timestamp', 'responsereason'
    COL_OPERATOR = 'requestrelatedpartyname'  # Campo del operador
    
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

    # APLICAR FILTRO DE OPERADOR (después del filtro de fecha)
    if operador_filtro == "ETB":
        df_filtrado = df_filtrado[df_filtrado[COL_OPERATOR].astype(str).str.strip().str.upper() == "ETB"]
    elif operador_filtro == "CLARO":
        df_filtrado = df_filtrado[df_filtrado[COL_OPERATOR].astype(str).str.strip().str.upper() != "ETB"]
    
    # Mostrar operador seleccionado en el dashboard
    st.sidebar.write(f"✅ Operador: **{operador_filtro}**")

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

    # --- BOTÓN PARA GENERAR ANÁLISIS (ANTES DE GENERAR) ---
    if 'generar_analisis' not in st.session_state:
        st.session_state.generar_analisis = False
    
    if not st.session_state.generar_analisis:
        st.write("---")
        col_btn1, col_btn2 = st.columns([2, 3])
        
        with col_btn1:
            if st.button("🚀 GENERAR ANÁLISIS COMPLETO", use_container_width=True, type="primary"):
                st.session_state.generar_analisis = True
                st.rerun()
        
        with col_btn2:
            st.info("⚙️ Ajusta fechas y operador, luego haz clic para generar el análisis con IA")

    # --- 3. GRÁFICA (siempre visible) ---
    st.write("---")
    fig, ax1 = plt.subplots(figsize=(16, 7))
    bars = ax1.bar(resumen[COL_TRANS], resumen['Total'], color='#A9A9A9', alpha=0.7, label='Q Transacciones')
    ax1.set_ylabel('Volumen de Transacciones', fontweight='bold', color='#4F4F4F', fontsize=12)
    
    ax2 = ax1.twinx()
    line = ax2.plot(resumen[COL_TRANS], resumen['% Éxito'], color='#C00000', marker='o', linewidth=3, markersize=10, label='% Éxito (SLA)')
    ax2.set_ylabel('% Éxito (SLA)', color='#C00000', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 120) 

    # ROTACIÓN DE ETIQUETAS PARA EVITAR SOLAPAMIENTO
    ax1.set_xticklabels(resumen[COL_TRANS], rotation=45, ha='right', fontsize=10)
    
    # AJUSTAR ESPACIOS PARA LAS ETIQUETAS
    plt.tight_layout()

    max_y = resumen['Total'].max()
    for i, bar in enumerate(bars):
        proc = resumen[COL_TRANS][i]
        height = bar.get_height()
        if proc.lower() in ['activación', 'terminación']:
            ax1.text(bar.get_x() + bar.get_width()/2., height / 2, f'{int(height)}', ha='center', va='center', fontweight='bold', color='black')
        else:
            ax1.text(bar.get_x() + bar.get_width()/2., height + (max_y * 0.01), f'{int(height)}', ha='center', va='bottom', fontweight='bold', color='black')

    for i, val in enumerate(resumen['% Éxito']):
        ax2.annotate(f"{val}%", (resumen[COL_TRANS][i], resumen['% Éxito'][i]), textcoords="offset points", xytext=(0, 15), ha='center', color='#C00000', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C00000", alpha=0.9))

    plt.title("ANÁLISIS DE VOLUMETRÍA Y TASA DE ÉXITO", fontweight='bold', fontsize=16, pad=20)
    st.pyplot(fig)

    # --- 4. CONSOLIDADO CON IA (SOLO SI SE PRESIONA BOTÓN) ---
    if st.session_state.generar_analisis:
        st.write("---")
        st.markdown("### 📝 Consolidado de Operación")
        
        current_key = f"{uploaded_file.name}_{rango_input[0]}_{rango_input[1]}_{operador_filtro}" if len(rango_input) == 2 else f"{uploaded_file.name}_all_{operador_filtro}"
        
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
            
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            ax_pie.pie(top_errores.values, autopct='%1.1f%%', pctdistance=0.85, startangle=70, colors=plt.cm.Paired.colors, textprops={'fontsize': 9})
            ax_pie.set_title("Distribución de Top 5 Errores Críticos", fontsize=14)
            ax_pie.axis('equal')
            
            col_pie, col_table = st.columns([2, 1])
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

        # --- 6. RESUMEN EJECUTIVO ---
        st.write("---")
        st.markdown("### 📌 Resumen Ejecutivo del Período (Generado con IA)")

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

        resumen_ejecutivo = ""  # INICIALIZAR VARIABLE
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
            st.info("⚠️ Ingrese una API Key para generar el resumen ejecutivo. (Sin IA, usaremos plantilla estándar)")
            resumen_ejecutivo = f"""
- SLA Global: {e_gen:.2f}% de éxito
- Proceso crítico: {proceso_mas_fallas}
- Volumen procesado: {total_tx:,} transacciones
- Tasa de error: {pct_no_exitosas:.2f}%
            """

        # --- BOTÓN AL FINAL DEL INFORME ---
        st.write("---")
        col_btn_final1, col_btn_final2, col_btn_final3 = st.columns([2, 2, 2])
        
        with col_btn_final1:
            if st.button("🔄 GENERAR NUEVAMENTE", use_container_width=True, type="secondary"):
                st.session_state.generar_analisis = False
                st.rerun()
        
        with col_btn_final2:
            # 1. Definición única del nombre (evita el error de Pylance)
            fecha_reporte = rango_input[0].strftime("%d%m%Y") if len(rango_input) == 2 else "COMPLETO"
            nombre_reporte = f"Reporte_ATP_{operador_filtro}_{fecha_reporte}"
            
            # 2. Captura de las gráficas actuales
            img_barras_64 = fig_to_base64(fig)
            img_pie_64 = fig_to_base64(fig_pie) if not df_solo_errores.empty else ""

            # 3. HTML CLON de la interfaz de Streamlit
            html_content = f"""
            <!DOCTYPE html>
            <html lang="es">
            <head>
                <meta charset="UTF-8">
                <style>
                    /* Estilos Base de Streamlit */
                    body {{ 
                        font-family: "Source Sans Pro", sans-serif; 
                        background-color: #f0f2f6; 
                        color: #31333F; 
                        margin: 0; padding: 40px;
                    }}
                    .main-container {{ max-width: 1200px; margin: auto; }}
                    
                    /* Tarjetas Blancas con Sombra Suave (Idénticas a la App) */
                    .stCard {{ 
                        background: white; 
                        border-radius: 10px; 
                        padding: 25px; 
                        margin-bottom: 25px; 
                        box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 2px 0px;
                        border: 1px solid #e6e9ef;
                    }}

                    /* Encabezado */
                    .header {{ margin-bottom: 30px; }}
                    h1 {{ font-size: 2.2rem; margin: 0; font-weight: 700; }}
                    h3 {{ color: #5e6472; font-weight: 400; margin-top: 5px; }}
                    
                    /* KPIs */
                    .kpi-row {{ display: flex; gap: 20px; margin-bottom: 25px; }}
                    .kpi-card {{ 
                        background: white; border-radius: 10px; padding: 15px 25px; flex: 1;
                        border: 1px solid #e6e9ef; box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 2px 0px;
                    }}
                    .kpi-label {{ font-size: 0.8rem; color: #5e6472; text-transform: uppercase; font-weight: 600; }}
                    .kpi-value {{ font-size: 1.8rem; font-weight: 600; color: #31333F; }}

                    /* Tablas */
                    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
                    th {{ text-align: left; padding: 12px; border-bottom: 2px solid #f0f2f6; color: #5e6472; font-weight: 600; }}
                    td {{ padding: 12px; border-bottom: 1px solid #f0f2f6; }}
                    tr:nth-child(even) {{ background-color: #f8f9fb; }}

                    /* Imágenes */
                    .img-fluid {{ width: 100%; height: auto; border-radius: 4px; }}
                    
                    .resumen-exec {{ 
                        background-color: #f0f2f6; border-radius: 8px; padding: 20px; 
                        border-left: 5px solid #ff4b4b; /* Rojo Streamlit */
                    }}
                </style>
            </head>
            <body>
                <div class="main-container">
                    <div class="header">
                        <h1>📊 ESTADO TRANSACCIONES SISTEMAS TI</h1>
                        <h3>OPERACIÓN COMPARTIDA {OPERADOR_LABEL[operador_filtro]}</h3>
                        <p style="color: #a3a8b4;">Periodo: {periodo_str} | ID: {nombre_reporte}</p>
                    </div>

                    <div class="kpi-row">
                        <div class="kpi-card"><div class="kpi-label">Total Transacciones</div><div class="kpi-value">{t_gen:,}</div></div>
                        <div class="kpi-card"><div class="kpi-label">SLA Global Éxito</div><div class="kpi-value">{e_gen:.2f}%</div></div>
                        <div class="kpi-card"><div class="kpi-label">No Exitosas</div><div class="kpi-value">{100-e_gen:.2f}%</div></div>
                    </div>

                    <div class="stCard">
                        <h2 style="margin-top:0; font-size: 1.2rem;">📈 Análisis de Volumetría y Tasa de Éxito</h2>
                        <img src="data:image/png;base64,{img_barras_64}" class="img-fluid">
                    </div>

                    <div class="stCard">
                        <h2 style="margin-top:0; font-size: 1.2rem;">📝 Consolidado de Operación</h2>
                        <table>
                            <thead>
                                <tr><th>Proceso</th><th>Total</th><th>Exitosas</th><th>% Éxito</th><th>Análisis IA</th></tr>
                            </thead>
                            <tbody>
                                {"".join([f"<tr><td><b>{row[COL_TRANS]}</b></td><td>{row['Total']}</td><td>{row['Exitosas']}</td><td>{row['% Éxito']:.1f}%</td><td>{row.get('Causal No Exitosa (Comentario)', 'N/A')}</td></tr>" for _, row in st.session_state['resumen_df'].iterrows()])}
                            </tbody>
                        </table>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div class="stCard">
                            <h2 style="margin-top:0; font-size: 1.2rem;">🚩 Top Errores Críticos</h2>
                            <img src="data:image/png;base64,{img_pie_64}" class="img-fluid">
                        </div>
                        <div class="stCard">
                            <h2 style="margin-top:0; font-size: 1.2rem;">📌 Resumen Ejecutivo</h2>
                            <div class="resumen-exec">
                                <ul style="margin:0; padding-left:20px;">
                                    {"".join([f"<li>{l.strip()}</li>" for l in resumen_ejecutivo.split(chr(10)) if l.strip()])}
                                </ul>
                            </div>
                        </div>
                    </div>

                    <div style="text-align: center; color: #a3a8b4; font-size: 11px; margin-top: 40px;">
                        Este documento es una copia fiel del reporte generado en ATP Dashboard TI.
                    </div>
                </div>
            </body>
            </html>
            """
               
            st.download_button(
                label="📥 Descargar Reporte HTML",
                data=html_content,
                file_name=f"{nombre_reporte}_{pd.Timestamp.now().strftime('%d%m%Y_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col_btn_final3:
            st.info("📊 Descarga o regenera el análisis")

else:
    st.info("👋 Por favor, carga el archivo CSV para comenzar.")