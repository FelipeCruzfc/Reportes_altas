import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from PIL import Image
from datetime import datetime, timedelta
import numpy as np

# Configuración
st.set_page_config(page_title="ATP - Análisis de Timeouts", layout="wide")

# ================================
# FUNCIONES AUXILIARES
# ================================

def limpiar_encabezado(df):
    nuevos_nombres = {col: re.sub(r'[^a-zA-Z0-9_]', '', col).lower() for col in df.columns}
    return df.rename(columns=nuevos_nombres)

def traducir_error(codigo, razon=None):
    if codigo == "200 TODO OK":
        return "200 TODO OK"
    descripcion = codigo
    
    # Códigos específicos
    if codigo == "100-5":
        return "100-5: REQUEST_TIME_OUT"
    if codigo == "100-4":
        return "100-4: Sin respuesta del servicio externo"
    if codigo == "200-46" and razon:
        if "debido a" in razon:
            razon_limpia = razon.split("debido a")[0].strip()
        return f"200-46: {razon_limpia}"
    
    if razon and razon.strip() and razon.lower() not in ['nan', 'none', '']:
        descripcion = f"{codigo}: {razon.strip()}"
    
    return descripcion

def obtener_franja_horaria(hora):
    """Clasifica la hora en franjas"""
    if 0 <= hora < 6:
        return "Madrugada (00-06h)"
    elif 6 <= hora < 12:
        return "Mañana (06-12h)"
    elif 12 <= hora < 18:
        return "Tarde (12-18h)"
    else:
        return "Noche (18-24h)"

# ================================
# HEADER
# ================================

ruta_logo = r"C:\Users\FelipeCruz\OneDrive - Andean Telecom Partners\Escritorio\Reportes_ATP\logo_ATP.png"

header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    try:
        img = Image.open(ruta_logo)
        st.image(img, width=180)
    except:
        st.write("⚠️ Logo ATP")

with header_col2:
    st.markdown("<h1 style='margin-bottom: 0;'>⏱️ ANÁLISIS DE TIMEOUTS - SISTEMAS TI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: gray; margin-top: 0px;'>Diagnóstico de Latencia y Efectos Cascada</h3>", unsafe_allow_html=True)

# ================================
# SIDEBAR - CARGA Y FILTROS
# ================================

st.sidebar.title("Configuración")
uploaded_file = st.sidebar.file_uploader("📂 Cargar CSV de Kibana", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, encoding='latin-1')
    df_clean = limpiar_encabezado(df_raw)
    
    # Filtro de operador
    operador_filtro = st.sidebar.selectbox(
        "🏢 Filtrar por Operador",
        ["Ambos", "CLARO", "ETB"]
    )
    
    # Mapeo de procesos
    MAPEO = {
        'addont': 'Activación', 
        'addservice': 'Reservas',
        'changeont': 'Cambio ONT', 
        'deleteont': 'Terminación',
        'neighborsquery': 'Soporte N1 Vecinos', 
        '-': 'Error de solicitud', 
        'activateservice': 'Reconexión',
        'bandwidthquery': 'Ancho de banda', 
        'signallevelquery': 'Consulta niveles', 
        'statusquery': 'Estado servicio',
        'suspendservice': 'Suspensión', 
        'deleteservice': 'Eliminación servicio'
    }
    
    COL_TRANS = 'requestdescription'
    COL_CODE = 'responsecode'
    COL_TIME = 'timestamp'
    COL_REASON = 'responsereason'
    COL_OPERATOR = 'requestrelatedpartyname'
    
    # Limpieza inicial
    df_clean[COL_TRANS] = df_clean[COL_TRANS].astype(str).str.strip().str.lower().replace(MAPEO)
    df_clean[COL_CODE] = df_clean[COL_CODE].astype(str).str.strip().replace(['nan', 'NaN', 'None', ''], '-')
    df_clean[COL_CODE] = df_clean[COL_CODE].apply(lambda x: "200 TODO OK" if x in ['.', '-'] else x)
    df_clean[COL_CODE] = df_clean.apply(lambda row: traducir_error(row[COL_CODE], row.get(COL_REASON)), axis=1)
    
    # Procesar timestamp
    if COL_TIME in df_clean.columns:
        df_clean[COL_TIME] = pd.to_datetime(
            df_clean[COL_TIME].astype(str).str.replace(' @ ', ' '), 
            errors='coerce'
        )
        df_clean = df_clean.dropna(subset=[COL_TIME])
        
        # Extraer componentes temporales
        df_clean['fecha'] = df_clean[COL_TIME].dt.date
        df_clean['hora'] = df_clean[COL_TIME].dt.hour
        df_clean['dia_semana'] = df_clean[COL_TIME].dt.day_name()
        df_clean['franja_horaria'] = df_clean['hora'].apply(obtener_franja_horaria)
        
        # Filtro de fecha
        rango_input = st.sidebar.date_input(
            "📅 Filtrar Periodo", 
            [df_clean[COL_TIME].min().date(), df_clean[COL_TIME].max().date()]
        )
        
        if len(rango_input) == 2:
            df_filtrado = df_clean[
                (df_clean[COL_TIME].dt.date >= rango_input[0]) & 
                (df_clean[COL_TIME].dt.date <= rango_input[1])
            ].copy()
            periodo_str = f"{rango_input[0]} al {rango_input[1]}"
        else:
            df_filtrado = df_clean.copy()
            periodo_str = "Seleccione un rango"
    else:
        df_filtrado = df_clean.copy()
        periodo_str = "N/A"
    
    # Aplicar filtro de operador
    if operador_filtro == "ETB":
        df_filtrado = df_filtrado[df_filtrado[COL_OPERATOR].astype(str).str.strip().str.upper() == "ETB"]
    elif operador_filtro == "CLARO":
        df_filtrado = df_filtrado[df_filtrado[COL_OPERATOR].astype(str).str.strip().str.upper() != "ETB"]
    
    st.sidebar.success(f"✅ Datos cargados: {len(df_filtrado):,} registros")
    
    # ================================
    # IDENTIFICAR TIMEOUTS
    # ================================
    
    # Puedes expandir esta lista si hay más códigos de timeout
    CODIGOS_TIMEOUT = [
        "100-5: REQUEST_TIME_OUT",
        "100-4: Sin respuesta del servicio externo"
    ]
    
    df_filtrado['es_timeout'] = df_filtrado[COL_CODE].isin(CODIGOS_TIMEOUT)
    df_timeouts = df_filtrado[df_filtrado['es_timeout']].copy()
    
    # ================================
    # KPIs PRINCIPALES
    # ================================
    
    total_transacciones = len(df_filtrado)
    total_timeouts = len(df_timeouts)
    pct_timeouts = (total_timeouts / total_transacciones * 100) if total_transacciones > 0 else 0
    
    # Calcular cuántos errores ocurren DESPUÉS de timeouts (efecto cascada)
    # Esto es una aproximación: errores en los siguientes 5 minutos del mismo proceso
    df_timeouts_sorted = df_timeouts.sort_values(COL_TIME)
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Transacciones", f"{total_transacciones:,}")
    
    with col2:
        st.metric("⏱️ Total Timeouts", f"{total_timeouts:,}")
    
    with col3:
        st.metric("📈 % de Timeouts", f"{pct_timeouts:.2f}%")
    
    with col4:
        st.write("**Período Analizado**")
        st.caption(periodo_str)
    
    # ================================
    # ANÁLISIS TEMPORAL
    # ================================
    
    st.markdown("---")
    st.markdown("### 📅 Análisis Temporal de Timeouts")
    
    tab1, tab2, tab3 = st.tabs(["Por Hora del Día", "Por Día de la Semana", "Tendencia Diaria"])
    
    with tab1:
        # Gráfico de timeouts por hora
        timeouts_por_hora = df_timeouts.groupby('hora').size().reindex(range(24), fill_value=0)
        
        fig1, ax1 = plt.subplots(figsize=(14, 5))
        bars = ax1.bar(timeouts_por_hora.index, timeouts_por_hora.values, 
                       color='#E74C3C', alpha=0.7, edgecolor='white')
        
        # Resaltar horas pico
        max_hora = timeouts_por_hora.idxmax()
        bars[max_hora].set_color('#C0392B')
        bars[max_hora].set_alpha(1)
        
        ax1.set_xlabel('Hora del Día (0-23h)', fontweight='bold')
        ax1.set_ylabel('Cantidad de Timeouts', fontweight='bold')
        ax1.set_title('Distribución de Timeouts por Hora del Día', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_xticks(range(24))
        
        # Anotar hora pico
        ax1.annotate(f'Hora Pico\n{timeouts_por_hora[max_hora]} timeouts',
                    xy=(max_hora, timeouts_por_hora[max_hora]),
                    xytext=(max_hora, timeouts_por_hora[max_hora] * 1.15),
                    ha='center',
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", fc="#C0392B", ec="white", alpha=0.9),
                    color='white',
                    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=2))
        
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
        
        # Tabla de top horas
        col_tab1, col_tab2 = st.columns(2)
        
        with col_tab1:
            st.markdown("**🔴 Top 5 Horas con Más Timeouts**")
            top_horas = timeouts_por_hora.nlargest(5).reset_index()
            top_horas.columns = ['Hora', 'Cantidad']
            top_horas['%'] = (top_horas['Cantidad'] / total_timeouts * 100).round(1)
            st.dataframe(top_horas, hide_index=True, use_container_width=True)
        
        with col_tab2:
            st.markdown("**✅ Top 5 Horas con Menos Timeouts**")
            bottom_horas = timeouts_por_hora.nsmallest(5).reset_index()
            bottom_horas.columns = ['Hora', 'Cantidad']
            bottom_horas['%'] = (bottom_horas['Cantidad'] / total_timeouts * 100).round(1) if total_timeouts > 0 else 0
            st.dataframe(bottom_horas, hide_index=True, use_container_width=True)
    
    with tab2:
        # Análisis por día de la semana
        orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dias_español = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        
        timeouts_por_dia = df_timeouts['dia_semana'].value_counts().reindex(orden_dias, fill_value=0)
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        bars2 = ax2.bar(range(7), timeouts_por_dia.values, color='#3498DB', alpha=0.7, edgecolor='white')
        
        # Resaltar día con más timeouts
        max_dia_idx = timeouts_por_dia.values.argmax()
        bars2[max_dia_idx].set_color('#2874A6')
        bars2[max_dia_idx].set_alpha(1)
        
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(dias_español, rotation=45, ha='right')
        ax2.set_ylabel('Cantidad de Timeouts', fontweight='bold')
        ax2.set_title('Distribución de Timeouts por Día de la Semana', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    
    with tab3:
        # Tendencia diaria
        timeouts_diarios = df_timeouts.groupby('fecha').size().reset_index()
        timeouts_diarios.columns = ['Fecha', 'Cantidad']
        
        fig3, ax3 = plt.subplots(figsize=(14, 5))
        ax3.plot(timeouts_diarios['Fecha'], timeouts_diarios['Cantidad'], 
                marker='o', linewidth=2, markersize=6, color='#E67E22')
        ax3.fill_between(timeouts_diarios['Fecha'], timeouts_diarios['Cantidad'], 
                         alpha=0.3, color='#E67E22')
        
        # Línea de promedio
        promedio = timeouts_diarios['Cantidad'].mean()
        ax3.axhline(y=promedio, color='#C0392B', linestyle='--', linewidth=2, 
                   label=f'Promedio: {promedio:.0f}')
        
        ax3.set_xlabel('Fecha', fontweight='bold')
        ax3.set_ylabel('Cantidad de Timeouts', fontweight='bold')
        ax3.set_title('Evolución Diaria de Timeouts', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='both', alpha=0.3, linestyle='--')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
        
        # Estadísticas
        st.markdown("**📊 Estadísticas de la Serie Temporal:**")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Promedio Diario", f"{promedio:.0f}")
        with col_stat2:
            st.metric("Día con Más Timeouts", f"{timeouts_diarios['Cantidad'].max()}")
        with col_stat3:
            st.metric("Día con Menos Timeouts", f"{timeouts_diarios['Cantidad'].min()}")
        with col_stat4:
            std_dev = timeouts_diarios['Cantidad'].std()
            st.metric("Desviación Estándar", f"{std_dev:.1f}")
    
    # ================================
    # ANÁLISIS POR PROCESO
    # ================================
    
    st.markdown("---")
    st.markdown("### 🎯 Timeouts por Proceso")
    
    timeouts_por_proceso = df_timeouts[COL_TRANS].value_counts().reset_index()
    timeouts_por_proceso.columns = ['Proceso', 'Timeouts']
    timeouts_por_proceso['% del Total Timeouts'] = (
        timeouts_por_proceso['Timeouts'] / total_timeouts * 100
    ).round(1)
    
    # Calcular tasa de timeout por proceso (timeouts / total transacciones del proceso)
    total_por_proceso = df_filtrado[COL_TRANS].value_counts()
    timeouts_por_proceso['Total Transacciones'] = timeouts_por_proceso['Proceso'].map(total_por_proceso)
    timeouts_por_proceso['Tasa de Timeout (%)'] = (
        timeouts_por_proceso['Timeouts'] / timeouts_por_proceso['Total Transacciones'] * 100
    ).round(2)
    
    col_proceso1, col_proceso2 = st.columns([1, 1.5])
    
    with col_proceso1:
        st.dataframe(
            timeouts_por_proceso,
            column_config={
                "% del Total Timeouts": st.column_config.NumberColumn(format="%.1f%%"),
                "Tasa de Timeout (%)": st.column_config.NumberColumn(format="%.2f%%")
            },
            hide_index=True,
            use_container_width=True
        )
    
    with col_proceso2:
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        
        # Gráfico de barras horizontal
        y_pos = range(len(timeouts_por_proceso))
        bars4 = ax4.barh(y_pos, timeouts_por_proceso['Timeouts'], color='#9B59B6', alpha=0.8)
        
        # Colorear barra con mayor tasa
        max_tasa_idx = timeouts_por_proceso['Tasa de Timeout (%)'].idxmax()
        bars4[max_tasa_idx].set_color('#6C3483')
        
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(timeouts_por_proceso['Proceso'])
        ax4.set_xlabel('Cantidad de Timeouts', fontweight='bold')
        ax4.set_title('Distribución de Timeouts por Proceso', fontsize=12, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Anotar valores
        for i, (bar, val) in enumerate(zip(bars4, timeouts_por_proceso['Timeouts'])):
            ax4.text(val, bar.get_y() + bar.get_height()/2, f' {val}', 
                    va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
    
    # ================================
    # HEATMAP: TIMEOUTS POR HORA Y PROCESO
    # ================================
    
    st.markdown("---")
    st.markdown("### 🔥 Mapa de Calor: Timeouts por Hora y Proceso")
    
    pivot_heatmap = df_timeouts.groupby([COL_TRANS, 'hora']).size().unstack(fill_value=0)
    
    if not pivot_heatmap.empty:
        fig5, ax5 = plt.subplots(figsize=(16, 6))
        
        sns.heatmap(pivot_heatmap, annot=True, fmt='d', cmap='YlOrRd', 
                   linewidths=0.5, ax=ax5, cbar_kws={'label': 'Cantidad de Timeouts'})
        
        ax5.set_xlabel('Hora del Día', fontweight='bold')
        ax5.set_ylabel('Proceso', fontweight='bold')
        ax5.set_title('Concentración de Timeouts por Proceso y Hora', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()
        
        st.info("💡 **Insight:** Este mapa ayuda a identificar patrones específicos de timeouts por proceso y horario")
    
    # ================================
    # ANÁLISIS DE ERRORES RELACIONADOS
    # ================================
    
    st.markdown("---")
    st.markdown("### 🔗 Análisis de Errores Post-Timeout (Efecto Cascada)")
    
    st.info("""
    **Metodología:** Se analizan los errores que ocurren en los 5 minutos posteriores a cada timeout 
    en el mismo proceso, para identificar posibles efectos cascada.
    """)
    
    # Crear ventana temporal de 5 minutos después de cada timeout
    errores_post_timeout = []
    
    for idx, timeout_row in df_timeouts.iterrows():
        tiempo_timeout = timeout_row[COL_TIME]
        proceso_timeout = timeout_row[COL_TRANS]
        
        # Buscar errores en los siguientes 5 minutos del mismo proceso
        ventana_fin = tiempo_timeout + timedelta(minutes=5)
        
        errores_ventana = df_filtrado[
            (df_filtrado[COL_TRANS] == proceso_timeout) &
            (df_filtrado[COL_TIME] > tiempo_timeout) &
            (df_filtrado[COL_TIME] <= ventana_fin) &
            (df_filtrado[COL_CODE] != "200 TODO OK")
        ]
        
        if not errores_ventana.empty:
            for _, error_row in errores_ventana.iterrows():
                errores_post_timeout.append({
                    'Timeout_Time': tiempo_timeout,
                    'Error_Time': error_row[COL_TIME],
                    'Proceso': proceso_timeout,
                    'Codigo_Error': error_row[COL_CODE],
                    'Minutos_Despues': (error_row[COL_TIME] - tiempo_timeout).total_seconds() / 60
                })
    
    if errores_post_timeout:
        df_cascada = pd.DataFrame(errores_post_timeout)
        
        col_cascada1, col_cascada2 = st.columns(2)
        
        with col_cascada1:
            st.metric("🔗 Errores Relacionados Detectados", len(df_cascada))
            st.metric("📊 Timeouts con Errores Subsecuentes", df_cascada['Timeout_Time'].nunique())
            
            # Top errores subsecuentes
            st.markdown("**Top Errores Post-Timeout:**")
            top_errores_cascada = df_cascada['Codigo_Error'].value_counts().head(5).reset_index()
            top_errores_cascada.columns = ['Código Error', 'Frecuencia']
            st.dataframe(top_errores_cascada, hide_index=True, use_container_width=True)
        
        with col_cascada2:
            # Distribución temporal de errores post-timeout
            fig6, ax6 = plt.subplots(figsize=(8, 5))
            
            ax6.hist(df_cascada['Minutos_Despues'], bins=20, color='#E74C3C', alpha=0.7, edgecolor='black')
            ax6.set_xlabel('Minutos después del Timeout', fontweight='bold')
            ax6.set_ylabel('Cantidad de Errores', fontweight='bold')
            ax6.set_title('Distribución Temporal de Errores Post-Timeout', fontsize=11, fontweight='bold')
            ax6.grid(axis='y', alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            st.pyplot(fig6)
            plt.close()
        
        # Detalle expandible
        with st.expander("📋 Ver Detalle de Errores Post-Timeout"):
            df_cascada_display = df_cascada.copy()
            df_cascada_display['Timeout_Time'] = df_cascada_display['Timeout_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_cascada_display['Error_Time'] = df_cascada_display['Error_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_cascada_display['Minutos_Despues'] = df_cascada_display['Minutos_Despues'].round(2)
            
            st.dataframe(df_cascada_display, use_container_width=True, height=400)
    
    else:
        st.success("✅ No se detectaron errores subsecuentes inmediatos (dentro de 5 min) después de los timeouts")
    
    # ================================
    # RESUMEN Y RECOMENDACIONES
    # ================================
    
    st.markdown("---")
    st.markdown("### 📋 Resumen Ejecutivo")
    
    # Identificar proceso más afectado
    proceso_critico = timeouts_por_proceso.iloc[0]['Proceso'] if not timeouts_por_proceso.empty else "N/A"
    tasa_critica = timeouts_por_proceso.iloc[0]['Tasa de Timeout (%)'] if not timeouts_por_proceso.empty else 0
    
    # Identificar hora pico
    hora_pico_val = timeouts_por_hora.idxmax()
    timeouts_hora_pico = timeouts_por_hora.max()
    
    st.markdown(f"""
    - **Impacto General:** {pct_timeouts:.2f}% de todas las transacciones terminaron en timeout ({total_timeouts:,} de {total_transacciones:,})
    - **Proceso Crítico:** **{proceso_critico}** presenta la mayor tasa de timeout ({tasa_critica:.2f}%)
    - **Hora Pico de Timeouts:** **{hora_pico_val}:00h** con {timeouts_hora_pico} timeouts
    - **Efecto Cascada:** {len(df_cascada) if errores_post_timeout else 0} errores detectados dentro de 5 min post-timeout
    """)
    
    # Recomendaciones basadas en datos
    st.markdown("### 💡 Recomendaciones")
    
    recomendaciones = []
    
    if pct_timeouts > 5:
        recomendaciones.append("🔴 **CRÍTICO:** Tasa de timeout >5%. Se requiere revisión urgente de infraestructura.")
    
    if timeouts_hora_pico > total_timeouts * 0.2:
        recomendaciones.append(f"⚠️ La hora {hora_pico_val}:00h concentra más del 20% de los timeouts. Considerar escalamiento horizontal en esa franja.")
    
    if tasa_critica > 10:
        recomendaciones.append(f"🎯 El proceso **{proceso_critico}** tiene tasa de timeout >10%. Revisar arquitectura específica de este flujo.")
    
    if errores_post_timeout:
        recomendaciones.append("🔗 Se detectó efecto cascada. Implementar circuit breakers y políticas de retry inteligentes.")
    
    if not recomendaciones:
        recomendaciones.append("✅ Los niveles de timeout están dentro de rangos aceptables.")
    
    for rec in recomendaciones:
        st.markdown(f"- {rec}")
    
else:
    st.info("👋 Por favor, carga el archivo CSV para comenzar el análisis de timeouts")