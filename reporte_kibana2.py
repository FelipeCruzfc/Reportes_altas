import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from PIL import Image
import os
import time
import io
import matplotlib.dates as mdates
import base64
from datetime import datetime

def fig_to_base64(fig):
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    return base64.b64encode(img_buf.read()).decode('utf-8')


# Nota: Se eliminó la dependencia a IA (Gemini). Generación de comentarios ahora es local.

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
    if codigo == "100-4":
        return "100-4: Sin respuesta del servicio externo"
    if codigo == "200-2":
        return "200-2: Error inesperado"



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
        if codigo == "200-42" and "no pueden" in razon_limpia:
            razon_limpia = razon_limpia.split("no pueden")[0].strip()
        descripcion = f"{codigo}: {razon_limpia}"
        

    return descripcion

# ================================
# Clasificación para SLA TÉCNICO
# ================================
def clasificar_sla(codigo_traducido):
    """
    Devuelve:
    OK_TECNICO → funcionó el sistema (incluye rechazos de negocio)
    FALLA_TECNICA → problema real de sistema/red
    """

    FALLAS_TECNICAS = [
        "100-5: REQUEST_TIME_OUT",
        "100-190: No se creo la ONT nueva; rollback ejecutado",
        "100-191: No se asociaron servicios a ONT nueva; rollback ejecutado",
        "200-46: Error ejecutando el comando ADD-ONU",
        "200-46: Error ejecutando el comando CFG-ONUBW",
        "200-46: Error ejecutando el comando DEL-ONU",
        "100-4: Sin respuesta del servicio externo",
        "200-88: No se encontro CVLAN relacionada para la OLT",
        "200-2: Error inesperado",
    ]

    if codigo_traducido in FALLAS_TECNICAS:
        return "FALLA_TECNICA"
    else:
        return "OK_TECNICO"


def limpiar_encabezado(df):
    nuevos_nombres = {col: re.sub(r'[^a-zA-Z0-9_]', '', col).lower() for col in df.columns}
    return df.rename(columns=nuevos_nombres)


def generar_comentario(api_key, errores, total_tx=None):
    # Generador local de comentario.
    # `errores` se espera como DataFrame con columnas ['Código','Cantidad'] o similar.
    try:
        if hasattr(errores, 'iterrows'):
            df = errores.copy()
        elif isinstance(errores, str):
            filas = []
            for line in errores.splitlines():
                parts = line.split(":")
                if len(parts) >= 2:
                    codigo = parts[0].strip()
                    m = re.search(r"(\d+)", line)
                    cantidad = int(m.group(1)) if m else 0
                    filas.append({'Código': codigo, 'Cantidad': cantidad})
            import pandas as _pd
            df = _pd.DataFrame(filas)
        else:
            # lista de tuplas
            import pandas as _pd
            df = _pd.DataFrame(errores, columns=['Código', 'Cantidad'])

        if df.empty:
            return 'Sin errores registrados'

        if total_tx is None:
            # No tenemos total de transacciones; usar suma de cantidades como fallback
            total_tx = int(df['Cantidad'].sum())

        total_fail = int(df['Cantidad'].sum())
        if total_fail == 0 or total_tx == 0:
            return 'Sin errores registrados'

        total_fail_pct = round(total_fail / total_tx * 100, 1)

        # Mapeo de abreviaciones y responsables
        def abreviar(codigo):
            cod = str(codigo).lower()
            if 'timeout' in cod or 'request_time_out' in cod or '100-5' in cod:
                return 'Timeout', 'Proveedor/Servicio externo'
            if 'ont' in cod or '200-68' in cod or '200-39' in cod:
                return 'ONT en inv.', 'Inventario/Operaciones'
            if 'vno' in cod or '200-15' in cod or '200-38' in cod:
                return 'No asoc. a VNO', 'Integración/Cliente'
            if 'error inesperado' in cod or '200-2' in cod:
                return 'Error inesper.', 'Sistema interno'
            if 'no se encontro cvlan' in cod or '200-88' in cod:
                return 'CVLAN falt.', 'Integración OLT'
            if 'cfg-onubw' in cod or 'add-onu' in cod or 'del-onu' in cod or '200-46' in cod:
                return 'Error cmd ONU', 'Sistema/Equipos'
            if '100-4' in cod:
                return 'Sin resp. svc ext.', 'Proveedor/Servicio externo'
            return codigo, 'Indeterminado'

        # Agregar columna con abreviacion y responsable
        df['Abrev'], df['Resp'] = zip(*df['Código'].map(lambda x: abreviar(x)))

        # Agregar por abreviatura (para evitar duplicados iguales)
        agg = df.groupby('Abrev', as_index=False).agg({'Cantidad': 'sum', 'Resp': lambda x: x.mode().iat[0] if not x.mode().empty else 'Indeterminado'})
        agg = agg.sort_values('Cantidad', ascending=False)

        # Seleccionar top2 abreviaturas distintas
        top = agg.head(2)
        top_list = top.to_dict('records')

        # Calcular porcentajes relativos al total_tx (para que la suma sea total_fail_pct)
        top_pcts = [round(r['Cantidad'] / total_tx * 100, 1) for r in top_list]
        sum_top_pcts = sum(top_pcts)
        others_pct = round(total_fail_pct - sum_top_pcts, 1)
        if others_pct < 0 and abs(others_pct) < 0.1:
            others_pct = 0.0

        # Construir texto evitando duplicados
        parts = []
        responsables = []
        for i, r in enumerate(top_list):
            parts.append(f"{r['Abrev']} ({top_pcts[i]}%)")
            responsables.append(r.get('Resp', 'Indeterminado'))

        if not parts:
            return 'Sin errores registrados'

        if len(parts) == 1:
            comentario = f"{parts[0]} y Otros ({others_pct}%)"
        else:
            comentario = f"{parts[0]}, {parts[1]} y Otros ({others_pct}%)"

        # Responsable probable: el más frecuente entre los top
        responsable_final = next((r for r in responsables if r and r != 'Indeterminado'), 'Indeterminado')
        comentario = f"{comentario} (Prob. responsable: {responsable_final})"

        return comentario
    except Exception:
        return 'Sin errores registrados'


def generar_resumen_ejecutivo_local(resumen_datos):
    # Crear un resumen ejecutivo local, 3-5 bullets basados en la info pasada.
    try:
        lines = []
        # Extraer campos de resumen_datos si es string
        t = None
        sla = None
        pct_no = None
        proc = None
        causas = None
        if isinstance(resumen_datos, str):
            for l in resumen_datos.splitlines():
                if 'Total de transacciones' in l:
                    m = re.search(r"(\d+)", l)
                    t = int(m.group(1)) if m else None
                if 'SLA global' in l or 'SLA global de éxito' in l:
                    m = re.search(r"(\d+\.?\d*)%", l)
                    sla = float(m.group(1)) if m else None
                if 'Porcentaje no exitosas' in l:
                    m = re.search(r"(\d+\.?\d*)%", l)
                    pct_no = float(m.group(1)) if m else None
                if 'Proceso con mayor impacto' in l:
                    proc = l.split(':', 1)[1].strip() if ':' in l else None
                if 'Principales causas' in l:
                    causas = l.split(':', 1)[1].strip() if ':' in l else None

        # Bullet 1: SLA
        if sla is not None:
            lines.append(f"SLA global de plataforma: {sla:.1f}%.")
        elif t is not None:
            lines.append(f"Volumen analizado: {t:,} transacciones.")

        # Bullet 2: Proceso crítico
        if proc:
            lines.append(f"Proceso con mayor impacto: {proc}.")

        # Bullet 3: Causas principales
        if causas:
            lines.append(f"Causas principales: {causas}.")

        # Bullet 4: Recomendación simple
        if pct_no is not None:
            if pct_no > 5:
                lines.append("Recomendación: Revisar integraciones y colas de retry.")
            else:
                lines.append("Recomendación: Monitoreo continuo; operativo dentro de umbrales.")

        if not lines:
            return "Sin observaciones relevantes."

        return "\n".join(lines[:5])
    except Exception:
        return "Sin observaciones relevantes."


header_col1, header_col2 = st.columns([1, 4])

with header_col1:
    try:
        img = Image.open("logo_ATP.png")
        st.image(img, width=180)
    except:
        st.write("⚠️ Logo ATP")



OPERADOR_LABEL = {
    "Ambos": "CLARO / ETB",
    "CLARO": "CLARO",
    "ETB": "ETB"
}



with header_col2:
    st.markdown("<h1 style='margin-bottom: 0; padding-left: 100px;'> ESTADO TRANSACCIONES SISTEMAS TI</h1>", unsafe_allow_html=True)

# Configuración en Sidebar
st.sidebar.title("Configuración")
# Generación local de comentarios (sin IA)
auto_gen_comments = st.sidebar.checkbox("Auto-generar comentarios (local)", value=True)
force_gen_button = st.sidebar.button("Generar comentarios ahora")

uploaded_file = st.sidebar.file_uploader("📂 Cargar CSV de Kibana", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, encoding='latin-1')
    df_clean = limpiar_encabezado(df_raw)
    # IMPRIMIR TODAS LAS COLUMNAS
    print("\n" + "="*60)
    print("COLUMNAS DISPONIBLES EN EL CSV:")
    print("="*60)
    for i, col in enumerate(df_clean.columns, 1):
        print(f"{i:2d}. '{col}'")
    print("="*60 + "\n")
    
    # Agregar filtro de operador
    operador_filtro = st.sidebar.selectbox(
        "🏢 Filtrar por Operador",
        ["Ambos", "CLARO", "ETB"]
    )
    with header_col2:
        st.markdown(f"<h3 style='color: gray; margin-top: 0px; padding-left: 200px;'>OPERACIÓN COMPARTIDA {OPERADOR_LABEL[operador_filtro]}</h3>", unsafe_allow_html=True)

    MAPEO = {
        'addont': 'Activación ONT', 'addservice': 'Activación Servicios',
        'changeont': 'Cambio ONT', 'deleteont': 'Terminación',
        'neighborsquery': 'Soporte N1 Vecinos', '-': 'Error de solicitud', 
        'activateservice': 'Reconexión',
        'bandwidthquery': 'Ancho de banda', 'signallevelquery': 'Consulta niveles', 
        'statusquery': 'Estado servicio',
        'suspendservice': 'Suspensión ', 'deleteservice': 'Eliminación servicio', 
        'check': 'Consulta Disponibilidad', 'add': 'Reserva', 
        'delete': 'Cancelación', 'query': 'Consulta reserva'
    }

    COL_TRANS, COL_CODE, COL_TIME, COL_REASON = 'requestrequestdescription', 'responseresponsefinalresponseresponsecode', 'timestamp', 'responseresponsefinalresponseresponsereason'
    COL_OPERATOR = 'requestrequestrelatedpartyname'
    
    # ✅ DEFINE LOS PROCESOS IMPORTANTES AQUÍ (pero NO uses df_filtrado todavía)
    PROCESOS_IMPORTANTES = [
        MAPEO['addont'], 
        MAPEO['addservice'], 
        MAPEO['deleteont'], 
        MAPEO['add'], 
        MAPEO['query'],
        MAPEO['check'],
    ]
    
    # ✅ PRIMERO LIMPIA df_clean
    df_clean[COL_TRANS] = df_clean[COL_TRANS].astype(str).str.strip().str.lower().replace(MAPEO)
    df_clean[COL_CODE] = df_clean[COL_CODE].astype(str).str.strip().replace(['nan', 'NaN', 'None', ''], '-')
    df_clean['Es_Exito'] = df_clean[COL_CODE].isin(['.', '-'])
    df_clean[COL_CODE] = df_clean[COL_CODE].apply(lambda x: "200 TODO OK" if x in ['.', '-'] else x)
    df_clean[COL_CODE] = df_clean.apply(lambda row: traducir_error(row[COL_CODE], row.get(COL_REASON)), axis=1)

    FALLAS_TECNICAS = [
        "100-5: REQUEST_TIME_OUT",
        "100-190: No se creo la ONT nueva; rollback ejecutado",
        "100-191: No se asociaron servicios a ONT nueva; rollback ejecutado",
        "200-46: Error ejecutando el comando ADD-ONU",
        "200-46: Error ejecutando el comando CFG-ONUBW",
        "200-46: Error ejecutando el comando DEL-ONU",
        "100-4: Sin respuesta del servicio externo",
        "200-88: No se encontro CVLAN relacionada para la OLT",
        "200-2: Error inesperado",
    ]

    df_clean["Es_Exito_Tecnico"] = ~df_clean[COL_CODE].isin(FALLAS_TECNICAS)

    # ✅ AHORA SÍ APLICA FILTROS DE FECHA (esto crea df_filtrado)
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

    # ✅ FILTRO DE OPERADOR
    if operador_filtro == "ETB":
        df_filtrado = df_filtrado[df_filtrado[COL_OPERATOR].astype(str).str.strip().str.upper() == "ETB"]
    elif operador_filtro == "CLARO":
        df_filtrado = df_filtrado[df_filtrado[COL_OPERATOR].astype(str).str.strip().str.upper() != "ETB"]
    
    st.sidebar.write(f"✅ Operador: **{operador_filtro}**")

    # ✅ AHORA SÍ CREA df_linea_tiempo (df_filtrado YA EXISTE)
    df_linea_tiempo = df_filtrado[df_filtrado[COL_TRANS].isin(PROCESOS_IMPORTANTES)].copy()
    
    # ... el resto de tu código continúa aquí ...
    # --- 2. KPIs ---
    resumen = df_filtrado.groupby(COL_TRANS).agg(
        Total=('Es_Exito_Tecnico', 'size'),
        Procesadas_correctamente=('Es_Exito_Tecnico', 'sum')
    ).reset_index()

    resumen['Fallas_técnicas_reales'] = resumen['Total'] - resumen['Procesadas_correctamente']
    resumen['% Sistema OK'] = (resumen['Procesadas_correctamente'] / resumen['Total'] * 100).round(1)
    resumen['% Falla técnica'] = (resumen['Fallas_técnicas_reales'] / resumen['Total'] * 100).round(1)

    t_gen = int(resumen['Total'].sum())
    e_gen = (resumen['Procesadas_correctamente'].sum() / t_gen * 100) if t_gen > 0 else 0

    st.markdown("""<style>[data-testid="stMetricValue"] { font-size: 26px !important; }</style>""", unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    k1.metric("TOTAL TRANSACCIONES", f"{t_gen:,}")
    k2.metric("SLA de Plataforma", f"{e_gen:.2f}%")
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

    # DICCIONARIO DE COLORES PARA CÓDIGOS (organizado por categorías)
    def obtener_color_codigo(codigo):
        """Asigna colores según el tipo de código"""
        
        # Verde para éxitos
        if '200 TODO OK' in codigo:
            return '#2ECC71'  # Verde éxito
        
        # Rojos/Naranjas para errores técnicos del sistema
        elif '100-5: REQUEST_TIME_OUT' in codigo:
            return '#E74C3C'  # Rojo timeout
        elif '100-4:' in codigo:
            return '#C0392B'  # Rojo oscuro
        elif '100-2: Falta parámetro' in codigo:
            return '#E67E22'  # Naranja
        elif '100-190:' in codigo or '100-191:' in codigo:
            return '#D35400'  # Naranja oscuro
        
        # Morados para errores de negocio/estado
        elif '200-40:' in codigo:  # Estados INACTIVO/CANCELADO/ACTIVO
            return '#9B59B6'  # Morado
        elif '200-42:' in codigo:  # Componentes CANCELADO
            return '#8E44AD'  # Morado oscuro
        elif '200-38:' in codigo:  # No asociado a VNO
            return '#6C3483'  # Morado muy oscuro
        
        # Azules para errores de inventario/ONT
        elif '200-68:' in codigo:  # ONT ya existe
            return '#3498DB'  # Azul
        elif '200-63:' in codigo:  # Reserva no activa
            return '#2980B9'  # Azul oscuro
        elif '200-39:' in codigo:  # Reserva ya tiene ONT
            return '#1F618D'  # Azul muy oscuro
        
        # Amarillos para errores de comandos
        elif '200-46:' in codigo:  # Error ejecutando comando
            return '#F39C12'  # Amarillo/Dorado
        elif '200-41:' in codigo:
            return '#D68910'  # Amarillo oscuro
        
        # Rosas para errores de reserva/servicio
        elif '200-15:' in codigo:  # No existe reserva
            return '#EC7063'  # Rosa
        elif '200-30:' in codigo:  # Reserva COMPLETADA
            return '#E74C3C'  # Rosa oscuro
        
        # Gris para "Otros"
        elif codigo == 'Otros':
            return '#BDC3C7'  # Gris claro
        
        elif '200-2:' in codigo:
            return '#E74C3C'  # Rojo error inesperado
        
        # Gris para otros códigos no categorizados
        else:
            return '#95A5A6'  # Gris

    # Preparar datos para barras apiladas
    pivot_data = df_filtrado.groupby([COL_TRANS, COL_CODE]).size().unstack(fill_value=0)

    # --- AGRUPAR CÓDIGOS POCO FRECUENTES EN "OTROS" ---
    umbral_otros = pivot_data.sum().sum() * 0.015  # 1.5% del total
    codigos_frecuentes = pivot_data.sum()[pivot_data.sum() >= umbral_otros].index.tolist()

    # Asegurar que '200 TODO OK' siempre esté incluido
    if '200 TODO OK' not in codigos_frecuentes and '200 TODO OK' in pivot_data.columns:
        codigos_frecuentes.insert(0, '200 TODO OK')

    # Crear columna "Otros" con códigos poco frecuentes
    codigos_otros = [col for col in pivot_data.columns if col not in codigos_frecuentes]
    if codigos_otros:
        pivot_data['Otros'] = pivot_data[codigos_otros].sum(axis=1)
        # Mantener solo los frecuentes + Otros
        pivot_data = pivot_data[codigos_frecuentes + ['Otros']]
    else:
        pivot_data = pivot_data[codigos_frecuentes]

    # Ordenar columnas: primero '200 TODO OK', luego por frecuencia, 'Otros' al final
    columnas_ordenadas = []
    if '200 TODO OK' in pivot_data.columns:
        columnas_ordenadas.append('200 TODO OK')

    # Agregar el resto ordenado por total descendente (excepto 'Otros')
    otras_columnas = [col for col in pivot_data.columns if col not in ['200 TODO OK', 'Otros']]
    if otras_columnas:
        suma_columnas = pivot_data[otras_columnas].sum().sort_values(ascending=False)
        columnas_ordenadas.extend(suma_columnas.index.tolist())

    # Agregar 'Otros' al final si existe
    if 'Otros' in pivot_data.columns:
        columnas_ordenadas.append('Otros')

    pivot_data = pivot_data[columnas_ordenadas]

    # Crear figura
    fig, ax1 = plt.subplots(figsize=(16, 7))

    # Crear barras apiladas
    bottom = None
    bars_dict = {}
    max_y_total = pivot_data.sum(axis=1).max()

    for codigo in pivot_data.columns:
        color = obtener_color_codigo(codigo)
        
        bars = ax1.bar(
            pivot_data.index,
            pivot_data[codigo],
            bottom=bottom,
            label=codigo,
            color=color,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5
        )
        
        bars_dict[codigo] = bars
        
        # Agregar etiquetas en segmentos significativos
        for i, (bar, val) in enumerate(zip(bars, pivot_data[codigo])):
            if val > 0:
                height = bar.get_height()
                total_barra = pivot_data.sum(axis=1).iloc[i]
                porcentaje = (val / total_barra) * 100
                
                # Mostrar número si es >3% de esa barra Y el segmento es visible
                if porcentaje > 3 and height > max_y_total * 0.02:
                    y_pos = bar.get_y() + height / 2
                    
                    # Ajustar tamaño de fuente según espacio
                    if height > max_y_total * 0.08:
                        fontsize = 9
                    elif height > max_y_total * 0.05:
                        fontsize = 8
                    else:
                        fontsize = 7
                    
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2,
                        y_pos,
                        f'{int(val)}',
                        ha='center',
                        va='center',
                        fontsize=fontsize,
                        fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.25, linewidth=0)
                    )
        
        # Actualizar el bottom para apilar
        if bottom is None:
            bottom = pivot_data[codigo].copy()
        else:
            bottom = bottom + pivot_data[codigo]

    ax1.set_ylabel('Volumen de Transacciones', fontweight='bold', color='#4F4F4F', fontsize=12)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Totales encima de cada barra con mejor formato
    totales = pivot_data.sum(axis=1)
    max_y = totales.max()
    for i, (trans, total) in enumerate(totales.items()):
        ax1.text(
            i,
            total + (max_y * 0.015),
            f'{int(total):,}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=11,
            color='#2C3E50',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.9, linewidth=1.5)
        )

    # Eje secundario para % Sistema OK
    ax2 = ax1.twinx()
    line = ax2.plot(
        resumen[COL_TRANS],
        resumen['% Sistema OK'],
        color='#C00000',
        marker='o',
        linewidth=3,
        markersize=10,
        label='% Sistema OK',
        zorder=5
    )
    ax2.set_ylabel('% Sistema OK', color='#C00000', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 120)
    ax2.grid(False)

    # Anotaciones de porcentaje con mejor diseño
    for i, val in enumerate(resumen['% Sistema OK']):
        ax2.annotate(
            f"{val}%",
            (resumen[COL_TRANS][i], resumen['% Sistema OK'][i]),
            textcoords="offset points",
            xytext=(0, 12),
            ha='center',
            color='#C00000',
            fontweight='bold',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C00000", alpha=0.95, linewidth=2)
        )

    # Rotación de etiquetas en eje X
    ax1.set_xticks(range(len(pivot_data.index)))
    ax1.set_xticklabels(pivot_data.index, rotation=45, ha='right', fontsize=10)

    # Leyenda mejorada (solo códigos mostrados en el gráfico)
    handles, labels = ax1.get_legend_handles_labels()

    # Crear leyenda con mejor formato
    legend = ax1.legend(
        handles,
        labels,
        loc='center left',
        bbox_to_anchor=(1.12, 0.5),
        framealpha=0.97,
        fontsize=8.5,
        title='Códigos Principales',
        title_fontsize=10,
        edgecolor='#7F8C8D',
        fancybox=True,
        shadow=True
    )

    # Mejorar apariencia de la leyenda
    legend.get_title().set_fontweight('bold')

    plt.title(
        "ANÁLISIS DE VOLUMETRÍA Y COMPOSICIÓN POR CÓDIGO DE RESPUESTA", 
        fontweight='bold', 
        fontsize=16, 
        pad=20,
        color='#2C3E50'
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)



    # --- INFORMACIÓN ADICIONAL SOBRE CÓDIGOS AGRUPADOS ---
    if codigos_otros:
        with st.expander("ℹ️ Ver códigos agrupados en 'Otros'"):
            st.markdown("**Los siguientes códigos fueron agrupados en 'Otros' por baja frecuencia:**")
            
            # Crear tabla con los códigos agrupados
            otros_detalle = []
            for codigo in codigos_otros:
                total = df_filtrado[df_filtrado[COL_CODE] == codigo].shape[0]
                pct = (total / df_filtrado.shape[0]) * 100
                otros_detalle.append({
                    'Código': codigo,
                    'Cantidad': total,
                    '% del Total': f"{pct:.2f}%"
                })
            
            # Ordenar por cantidad
            otros_detalle = sorted(otros_detalle, key=lambda x: x['Cantidad'], reverse=True)
            
            for item in otros_detalle:
                st.markdown(f"• **{item['Código']}**: {item['Cantidad']} casos ({item['% del Total']})")
            
            st.info(f" Total de códigos en 'Otros': {len(codigos_otros)}")

    st.write("---")
    st.markdown("###  Evolución Temporal de Procesos Críticos")

    if not df_linea_tiempo.empty and COL_TIME in df_linea_tiempo.columns:
        
        # Agrupar por intervalos (ajusta según el rango de días)
        dias_en_rango = (df_linea_tiempo[COL_TIME].max() - df_linea_tiempo[COL_TIME].min()).days
        
        # Decidir intervalo automáticamente
        if dias_en_rango <= 1:
            intervalo = '5min'
            formato_fecha = '%H:%M'
            locator_interval = mdates.HourLocator(interval=1)
        elif dias_en_rango <= 7:
            intervalo = '30min'
            formato_fecha = '%d/%m %H:%M'
            locator_interval = mdates.HourLocator(interval=6)
        else:
            intervalo = '1H'
            formato_fecha = '%d/%m'
            locator_interval = mdates.DayLocator(interval=1)
        
        # Agrupar datos
        df_linea_tiempo['Intervalo'] = df_linea_tiempo[COL_TIME].dt.floor(intervalo)
        df_tiempo = df_linea_tiempo.groupby(['Intervalo', COL_TRANS]).size().reset_index(name='Count')
        
        # Crear figura
        fig_tiempo, ax_tiempo = plt.subplots(figsize=(16, 6))
        
        # Colores
        colores = {
            'Activación ONT': '#E74C3C',
            'Activación Servicios': '#9B59B6',
            'Terminación': '#3498DB',
            'Reserva': '#1ABC9C',
            'Consulta reserva': '#F39C12'
        }
        
        # Graficar cada proceso
        for proceso in PROCESOS_IMPORTANTES:
            df_proc = df_tiempo[df_tiempo[COL_TRANS] == proceso]
            if not df_proc.empty:
                ax_tiempo.plot(
                    df_proc['Intervalo'], 
                    df_proc['Count'],
                    label=proceso,
                    linewidth=1.5,
                    color=colores.get(proceso, '#95A5A6'),
                    alpha=0.8
                )
        
        # Configuración estilo Kibana
        ax_tiempo.set_xlabel(f'@timestamp per {intervalo}', fontsize=10, color='#666')
        ax_tiempo.set_ylabel('Count of records', fontsize=10, color='#666')
        ax_tiempo.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='#E0E0E0')
        ax_tiempo.set_facecolor('#FAFAFA')
        ax_tiempo.spines['top'].set_visible(False)
        ax_tiempo.spines['right'].set_visible(False)
        
        # Leyenda
        ax_tiempo.legend(
            loc='upper right',
            fontsize=9,
            frameon=True,
            framealpha=0.95,
            edgecolor='#CCC'
        )
        
        # Formato de fechas
        import matplotlib.dates as mdates
        ax_tiempo.xaxis.set_major_formatter(mdates.DateFormatter(formato_fecha))
        ax_tiempo.xaxis.set_major_locator(locator_interval)
        plt.xticks(rotation=45 if dias_en_rango > 1 else 0, ha='right' if dias_en_rango > 1 else 'center', fontsize=9)
        
        # Ajustar límites
        max_val = df_tiempo['Count'].max()
        ax_tiempo.set_ylim(0, max_val * 1.15)
        
        plt.tight_layout()
        st.pyplot(fig_tiempo)
        
        # GUARDAR PARA HTML
        img_tiempo_64 = fig_to_base64(fig_tiempo)
        plt.close(fig_tiempo)
    else:
        st.warning("⚠️ No hay datos para generar la línea de tiempo")
        img_tiempo_64 = ""

    # --- 4. CONSOLIDADO CON IA (SOLO SI SE PRESIONA BOTÓN) ---
    if st.session_state.generar_analisis:
        st.write("---")
        st.markdown("### Consolidado de Operación")
    
        current_key = f"{uploaded_file.name}_{rango_input[0]}_{rango_input[1]}_{operador_filtro}" if len(rango_input) == 2 else f"{uploaded_file.name}_all_{operador_filtro}"
        st.session_state.pop('resumen_df', None)

        
        if 'resumen_df' not in st.session_state or st.session_state.get('last_key') != current_key:
            # Decidir si se generan comentarios automáticamente o por botón (siempre local)
            do_generate = auto_gen_comments or force_gen_button
            if do_generate:
                with st.spinner("Generando comentarios localmente..."):
                    # Si ya existe un resumen previo, preservamos comentarios manuales
                    existing_comments = {}
                    if 'resumen_df' in st.session_state:
                        try:
                            tmp = st.session_state['resumen_df']
                            if COL_TRANS in tmp.columns and 'Causal No Exitosa (Comentario)' in tmp.columns:
                                existing_comments = tmp.set_index(COL_TRANS)['Causal No Exitosa (Comentario)'].to_dict()
                        except Exception:
                            existing_comments = {}

                    for idx in resumen.index:
                        proc = resumen.loc[idx, COL_TRANS]
                        # Si ya hay comentario manual, no sobrescribir
                        existing = existing_comments.get(proc, "")
                        if existing and str(existing).strip():
                            resumen.at[idx, 'Causal No Exitosa (Comentario)'] = existing
                            continue

                        df_proc = df_filtrado[df_filtrado[COL_TRANS] == proc]
                        df_err = df_proc[~df_proc["Es_Exito_Tecnico"]]  # solo fallas técnicas reales

                        if not df_err.empty:
                            errores_df = df_err[COL_CODE].value_counts().reset_index()
                            errores_df.columns = ['Código', 'Cantidad']
                            errores_df['%'] = (errores_df['Cantidad'] / len(df_proc) * 100).round(1)
                            try:
                                comentario = generar_comentario(None, errores_df, total_tx=len(df_proc))
                            except Exception:
                                comentario = ""

                            resumen.at[idx, 'Causal No Exitosa (Comentario)'] = comentario or ""
                        else:
                            resumen.at[idx, 'Causal No Exitosa (Comentario)'] = "Sin errores registrados"
            else:
                # No se generaron comentarios; mantenemos columna vacía para edición manual
                resumen['Causal No Exitosa (Comentario)'] = ""
            st.session_state['resumen_df'] = resumen
            st.session_state['last_key'] = current_key
            st.session_state['resumen_df'] = resumen
            st.session_state['last_key'] = current_key

        columnas_mostrar = [
            COL_TRANS,
            'Total',
            'Procesadas_correctamente',
            'Fallas_técnicas_reales',
            '% Sistema OK',
            '% Falla técnica',
            'Causal No Exitosa (Comentario)'
        ]

        
        if 'Causal No Exitosa (Comentario)' not in st.session_state['resumen_df'].columns:
            st.session_state['resumen_df']['Causal No Exitosa (Comentario)'] = ""
        
        df_editor = st.data_editor(
            st.session_state['resumen_df'][columnas_mostrar],
            column_config={
                "% Sistema OK": st.column_config.NumberColumn(format="%.1f%%"),
                "% Falla técnica": st.column_config.NumberColumn(format="%.1f%%"),
                "Causal No Exitosa (Comentario)": st.column_config.TextColumn("Comentarios", width="large")
            },
            hide_index=True, width='stretch'
        )
        st.session_state['resumen_df'] = df_editor

        # Mostrar fila de Totales debajo del consolidado para evidenciar métricas origen
        try:
            df_tot = st.session_state['resumen_df'].copy()
            total_T = int(df_tot['Total'].sum())
            proc_ok_sum = int(df_tot['Procesadas_correctamente'].sum())
            fallas_sum = int(df_tot['Fallas_técnicas_reales'].sum())
            pct_ok_overall = (proc_ok_sum / total_T * 100) if total_T > 0 else 0
            pct_falla_overall = (fallas_sum / total_T * 100) if total_T > 0 else 0

            import pandas as _pd
            df_tot_row = _pd.DataFrame([{
                COL_TRANS: 'Totales',
                'Total': total_T,
                'Procesadas_correctamente': proc_ok_sum,
                'Fallas_técnicas_reales': fallas_sum,
                '% Sistema OK': round(pct_ok_overall, 1),
                '% Falla técnica': round(pct_falla_overall, 1),
                'Causal No Exitosa (Comentario)': ''
            }])

            st.markdown("**Totales Consolidados**")
            st.table(df_tot_row)
        except Exception:
            pass

        st.write("---")


        st.markdown("### Comportamiento del Proceso")
        
        cols_grid = st.columns(2)
        for i, proc in enumerate(resumen[COL_TRANS].unique()):
            with cols_grid[i % 2]:
                st.markdown(f"**Análisis: {proc}**")
                df_proc = df_filtrado[df_filtrado[COL_TRANS] == proc]
                detalle = df_proc[COL_CODE].value_counts().reset_index()
                detalle.columns = ['Código / Descripción', 'Cantidad']
                detalle['%'] = (detalle['Cantidad'] / len(df_proc) * 100).map("{:.1f}%".format)
                st.table(detalle)
        
        # --- 5. DETALLES ---
        st.write("---")
        st.markdown("###  Top 5 Fallas Técnicas Reales")

        df_solo_errores = df_filtrado[~df_filtrado["Es_Exito_Tecnico"]]

        if not df_solo_errores.empty:
            top_errores = df_solo_errores[COL_CODE].value_counts().head(5)
            
            total_fallas = df_solo_errores.shape[0]
            total_tx = df_filtrado.shape[0]
            pct_fallas_global = (total_fallas / total_tx) * 100
            
            # Métricas principales en columnas
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Total Fallas Técnicas", f"{total_fallas:,}")
            with col_m2:
                st.metric("Impacto Operacional", f"{pct_fallas_global:.2f}%")
            with col_m3:
                st.metric("Tipos de Fallas", len(top_errores))
            
            st.write("")  # Espacio
            
            # Layout: Tabla a la izquierda, gráfico a la derecha
            col_table, col_pie = st.columns([1.2, 1])
            
            with col_table:
                st.markdown("** Detalle de Fallas:**")
                
                # Crear DataFrame para la tabla
                tabla_errores = []
                colores_usados = plt.cm.Set3.colors[:len(top_errores)]
                
                for idx, (error, cantidad) in enumerate(top_errores.items()):
                    pct = (cantidad / total_fallas) * 100
                    # Crear un cuadrito de color usando HTML
                    color_hex = '#{:02x}{:02x}{:02x}'.format(
                        int(colores_usados[idx][0]*255),
                        int(colores_usados[idx][1]*255),
                        int(colores_usados[idx][2]*255)
                    )
                    tabla_errores.append({
                        "Color": color_hex,
                        "Código": error,
                        "Cantidad": cantidad,
                        "% del Total": f"{pct:.1f}%"
                    })
                
                # Mostrar tabla con estilos
                for i, fila in enumerate(tabla_errores):
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 8px; padding: 8px; background-color: #f8f9fa; border-radius: 5px;">
                        <div style="width: 20px; height: 20px; background-color: {fila['Color']}; border-radius: 3px; margin-right: 10px; border: 1px solid #ddd;"></div>
                        <div style="flex: 1;">
                            <strong>{fila['Código']}</strong><br>
                            <small>{fila['Cantidad']} casos ({fila['% del Total']})</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_pie:
                # Gráfico de torta más pequeño y limpio
                fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
                
                def autopct_format(pct):
                    return f'{pct:.1f}%' if pct > 5 else ''
                
                wedges, texts, autotexts = ax_pie.pie(
                    top_errores.values,
                    autopct=autopct_format,
                    pctdistance=0.70,
                    startangle=90,
                    colors=plt.cm.Set3.colors[:len(top_errores)],
                    textprops={'fontsize': 9, 'weight': 'bold'}
                )
                
                ax_pie.set_title("Distribución de Fallas", fontsize=11, pad=10)
                ax_pie.axis('equal')
                
                st.pyplot(fig_pie)
                plt.close(fig_pie)
            
            # Insight adicional
            st.info(f" **Insight:** Las 5 principales fallas representan el {(top_errores.sum() / total_fallas * 100):.1f}% de todas las fallas técnicas.")

        else:
            st.success("✅ No se detectaron fallas técnicas en el período seleccionado.")


        # --- BOTÓN AL FINAL DEL INFORME ---
        st.write("---")
        col_btn_final1, col_btn_final2, col_btn_final3 = st.columns([2, 2, 2])
        
        with col_btn_final1:
            if st.button("🔄 GENERAR NUEVAMENTE", use_container_width=True, type="secondary"):
                st.session_state.generar_analisis = False
                st.rerun()
        
        with col_btn_final2:
            # --- 1. CARGAR LOGO LOCAL ---
            try:
                with open("logo_ATP.png", "rb") as f:
                    logo_img_64 = base64.b64encode(f.read()).decode()
                logo_src = f"data:image/png;base64,{logo_img_64}"
            except FileNotFoundError:
                logo_src = "https://www.atp-vcf.com/wp-content/uploads/2021/06/logo-ATP.png"

            # --- 2. CÁLCULO DE PERIODO Y RANGO ---
            if not df_filtrado.empty and COL_TIME in df_filtrado.columns:
                ts_inicio = df_filtrado[COL_TIME].min()
                ts_fin = df_filtrado[COL_TIME].max()
                periodo_detallado = f"Desde el {ts_inicio.strftime('%d/%m/%Y %H:%M:%S')} hasta el {ts_fin.strftime('%d/%m/%Y %H:%M:%S')}"
            else:
                periodo_detallado = "Periodo no definido"

            if len(rango_input) == 2:
                rango_texto = f"{rango_input[0].strftime('%d-%m-%Y')}_al_{rango_input[1].strftime('%d-%m-%Y')}"
            else:
                rango_texto = "Completo"

            # --- 3. PREPARAR DATOS PARA EL RESUMEN ---
            total_tx = df_filtrado.shape[0]
            pct_no_exitosas = 100 - e_gen
            proceso_mas_fallas = resumen.sort_values('Fallas_técnicas_reales', ascending=False).iloc[0][COL_TRANS] if not resumen.empty else "N/A"
            top_causas = df_solo_errores[COL_CODE].value_counts().head(3).index.tolist() if not df_solo_errores.empty else []

            resumen_datos_ia = f"""
            Total de transacciones: {total_tx}
            SLA global de éxito: {e_gen:.1f}%
            Porcentaje no exitosas: {pct_no_exitosas:.1f}%
            Proceso con mayor impacto: {proceso_mas_fallas}
            Principales causas: {', '.join(top_causas)}
            """

            # --- 4. GENERAR EL TEXTO DEL RESUMEN (local) ---
            resumen_ia = generar_resumen_ejecutivo_local(resumen_datos_ia)
            texto_resumen_html = "".join([f"<li style='margin-bottom:8px;'>{linea.strip()}</li>" for linea in resumen_ia.split("\n") if linea.strip()])

            # --- 5. PREPARAR GRÁFICAS Y TABLAS DE DETALLE ---
            img_barras_64 = fig_to_base64(fig)
            img_pie_64 = fig_to_base64(fig_pie) if not df_solo_errores.empty else ""

            # Tablas de errores por proceso (Las pequeñas de abajo)
            tablas_errores_html = ""
            for proc in resumen[COL_TRANS].unique():
                df_p = df_filtrado[df_filtrado[COL_TRANS] == proc]
                detalle_err = df_p[COL_CODE].value_counts().reset_index()
                detalle_err.columns = ['Error', 'Cant']
                detalle_err['%'] = (detalle_err['Cant'] / len(df_p) * 100).map("{:.1f}%".format)
                tablas_errores_html += f"""
                <div style="width: 48%; display: inline-block; vertical-align: top; margin-bottom: 20px; margin-right: 1%;">
                    <p style="font-weight: bold; font-size: 13px; margin: 5px 0; color: #31333F; border-left: 3px solid #C00000; padding-left: 8px;"> Análisis: {proc}</p>
                    <table style="font-size: 10px; width: 100%; border: 1px solid #eee;">
                        <thead><tr style="background: #f8f9fb;"><th>Error</th><th>Cant.</th><th>%</th></tr></thead>
                        <tbody>{"".join([f"<tr><td>{r['Error']}</td><td>{r['Cant']}</td><td>{r['%']}</td></tr>" for _, r in detalle_err.iterrows()])}</tbody>
                    </table>
                </div>
                """

            # Leyenda Top 5 CORREGIDA (Sincronizada con colores del gráfico de torta)
            top_5_detalle_html = ""
            if not df_solo_errores.empty:
                df_top_5 = df_solo_errores[COL_CODE].value_counts().head(5).reset_index()
                df_top_5.columns = ['Error', 'Frecuencia']
                colores_plotly = ["#7fcfad", "#ffffb3", "#bebada", "#fb8072", "#80b1d3"]
                
                leyenda_items = ""
                filas_tabla = ""
                for i, r in df_top_5.iterrows():
                    color = colores_plotly[i] if i < len(colores_plotly) else "#cccccc"
                    porcentaje = (r['Frecuencia'] / df_solo_errores.shape[0]) * 100
                    leyenda_items += f"""
                    <div style='font-size:12px; margin-bottom:5px; display:flex; align-items:center;'>
                        <span style='background-color:{color}; width:12px; height:12px; display:inline-block; margin-right:8px; border-radius:2px;'></span>
                        {r['Error']}
                    </div>"""
                    filas_tabla += f"<tr><td>{i+1}</td><td>{r['Error']}</td><td style='text-align:right;'>{r['Frecuencia']:,}</td><td style='text-align:right;'>{porcentaje:.1f}%</td></tr>"
                
                # Totales globales de fallas (para incluir en el HTML)
                total_fallas_html = df_solo_errores.shape[0]
                total_tx_html = df_filtrado.shape[0]
                pct_fallas_global_html = (total_fallas_html / total_tx_html * 100) if total_tx_html > 0 else 0

                top_5_detalle_html = f"""
                <div style="background: #f8f9fb; padding: 15px; border-radius: 8px; border: 1px solid #e6e9ef;">
                    <p style="font-weight: bold; font-size: 14px; margin: 0 0 6px 0;">Leyenda y Detalle Numérico:</p>
                    <p style="margin: 0 0 8px 0;"><b>Total fallas:</b> {total_fallas_html:,} ({pct_fallas_global_html:.1f}% del total transacciones)</p>
                    {leyenda_items}
                    <table style="font-size: 11px; width: 100%; margin-top: 15px; background: white;">
                        <thead><tr style="background: #eee;"><th>#</th><th>Código / Descripción</th><th>Cant.</th><th>%</th></tr></thead>
                        <tbody>{filas_tabla}</tbody>
                    </table>
                </div>
                """

            # --- 6. ENSAMBLAJE DEL HTML FINAL ---
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: 'Segoe UI', sans-serif; background-color: #f4f7f9; color: #333; padding: 20px; }}
                    .main-card {{ max-width: 1100px; margin: auto; background: white; padding: 35px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
                    .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #C00000; padding-bottom: 20px; }}
                    .kpi-container {{ display: flex; gap: 15px; margin: 25px 0; }}
                    .kpi-card {{ flex: 1; background: #fff; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #eee; }}
                    .kpi-val {{ font-size: 26px; font-weight: bold; color: #C00000; }}
                    .resumen-ejecutivo {{ background: #fff4f4; padding: 15px; border-left: 5px solid #C00000; border-radius: 4px; margin: 20px 0; font-size: 13px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-bottom: 10px; }}
                    th {{ text-align: left; padding: 10px; border-bottom: 2px solid #eee; font-size: 12px; background: #f8f9fb; }}
                    td {{ padding: 10px; border-bottom: 1px solid #eee; font-size: 11px; }}
                    .section-header {{ font-size: 18px; font-weight: bold; margin: 30px 0 15px 0; display: flex; align-items: center; }}
                    .red-line {{ width: 5px; height: 22px; background: #C00000; margin-right: 12px; }}
                </style>
            </head>
            <body>
            
                <div class="main-card">

                    <div class="header">

                        <img src="{logo_src}" style="height: 50px;">

                        <div style="text-align: right;">

                            <h1 style="margin:0; font-size: 18px;">ESTADO TRANSACCIONES SISTEMAS TI</h1>

                            <p style="margin:0; color: #C00000; font-weight: bold; font-size: 22px;">{OPERADOR_LABEL.get(operador_filtro, operador_filtro)}</p>

                        </div>

                    </div>



                    <div class="period-box">

                        <strong>DATOS ANALIZADOS:</strong> {periodo_detallado}

                    </div>



                    <div class="kpi-container">

                        <div class="kpi-card"><div>Transacciones</div><div class="kpi-val">{total_tx:,}</div></div>

                        <div class="kpi-card"><div>SLA Global</div><div class="kpi-val">{e_gen:.2f}%</div></div>

                        <div class="kpi-card"><div>Error Técnico</div><div class="kpi-val" style="color:#d35400;">{pct_no_exitosas:.2f}%</div></div>

                    </div>

                    
                    <div class="section-header"><div class="red-line"></div> 1. Análisis de Volumetría</div>
                    <img src="data:image/png;base64,{img_barras_64}" style="width: 100%; border-radius: 8px;">

                    <div class="section-header"><div class="red-line"></div> 2. Consolidado de Operación</div>
                    <table>
                        <thead>
                            <tr>
                                <th>Proceso</th>
                                <th>Total</th>
                                <th>Procesadas OK</th>
                                <th>Fallas Técnicas</th>
                                <th>% Sistema OK</th>
                                <th>% Falla</th>
                                <th>Comentarios</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join([f"""
                            <tr>
                                <td><b>{r[COL_TRANS]}</b></td>
                                <td>{int(r['Total']):,}</td>
                                <td>{int(r['Procesadas_correctamente']):,}</td>
                                <td>{int(r['Fallas_técnicas_reales']):,}</td>
                                <td><b>{r['% Sistema OK']:.1f}%</b></td>
                                <td>{r['% Falla técnica']:.1f}%</td>
                                <td>{r.get('Causal No Exitosa (Comentario)', 'N/A')}</td>
                            </tr>""" for _, r in st.session_state['resumen_df'].iterrows()])}
                        
                            <!-- Fila de Totales -->
                            {"""<tr style='font-weight:bold; background:#f7f9fb;'>
                                <td>Totales</td>
                                <td>{:,}</td>
                                <td>{:,}</td>
                                <td>{:,}</td>
                                <td><b>{:.1f}%</b></td>
                                <td>{:.1f}%</td>
                                <td></td>
                            </tr>""" .format(
                                int(st.session_state['resumen_df']['Total'].sum()),
                                int(st.session_state['resumen_df']['Procesadas_correctamente'].sum()),
                                int(st.session_state['resumen_df']['Fallas_técnicas_reales'].sum()),
                                (st.session_state['resumen_df']['Procesadas_correctamente'].sum() / st.session_state['resumen_df']['Total'].sum() * 100) if st.session_state['resumen_df']['Total'].sum() > 0 else 0,
                                (st.session_state['resumen_df']['Fallas_técnicas_reales'].sum() / st.session_state['resumen_df']['Total'].sum() * 100) if st.session_state['resumen_df']['Total'].sum() > 0 else 0
                            )}
                            </tbody>
                    </table>

                    <div class="section-header"><div class="red-line"></div> 3. Comportamiento del Proceso (Detalle)</div>
                    <div>{tablas_errores_html}</div>

                    <div style="display: flex; gap: 30px; margin-top: 20px; align-items: flex-start;">
                        <div style="flex: 1.2;">
                            <div class="section-header"><div class="red-line"></div> 4. Top 5 Errores Globales</div>
                            <img src="data:image/png;base64,{img_pie_64}" style="width: 100%;">
                        </div>
                        <div style="flex: 1; margin-top: 60px;">
                            {top_5_detalle_html}
                        </div>
                    </div>
                    
                    <p style="text-align: center; color: #999; font-size: 10px; margin-top: 30px;">
                        Reporte generado el {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Periodo: {periodo_detallado}
                    </p>
                </div>
            </body>
            </html>
            """

            # --- BOTÓN DE DESCARGA ---
            st.download_button(
                label="📥 Descargar Reporte Completo",
                data=html_content,
                file_name=f"Reporte_Final_{operador_filtro}_{rango_texto}.html",
                mime="text/html",
                use_container_width=True
            )
        # Justo después del botón "Descargar Reporte HTML"
        with col_btn_final3:
            if img_tiempo_64:  # Solo si hay gráfico
                html_solo_tiempo = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <style>
                        body {{ font-family: "Source Sans Pro", sans-serif; background: #f0f2f6; padding: 20px; }}
                        .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px; }}
                        h1 {{ color: #C00000; border-bottom: 3px solid #C00000; padding-bottom: 10px; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1> Evolución Temporal - Procesos</h1>
                        <p><b>Período:</b> {rango_texto} | <b>Operador:</b> {OPERADOR_LABEL[operador_filtro]}</p>
                        <img src="data:image/png;base64,{img_tiempo_64}" style="width: 100%; margin-top: 20px;">
                    </div>
                </body>
                </html>
                """
        
        st.download_button(
            label="📊 Descargar Solo Línea de Tiempo",
            data=html_solo_tiempo,
            file_name=f"LineaTiempo_{operador_filtro}_{rango_texto}.html",
            mime="text/html",
            use_container_width=True
        )

else:
    st.info("👋 Por favor, carga el archivo CSV para comenzar.")