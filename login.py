import streamlit as st
import requests
import pandas as pd
import urllib3
import os
from dotenv import load_dotenv
from datetime import datetime

# Cargar variables de entorno
load_dotenv()

BASE_URL = os.getenv("BASE_URL")
AUTH = (os.getenv("KIBANA_USER"), os.getenv("KIBANA_PASS"))

HEADERS = {
    "kbn-xsrf": "true",
    "Content-Type": "application/json"
}

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CAMPO_SUCIO = "response.response.finalResponse.Response.serviceQualificationItem.service.feature.featureCharacteristic.value.keyword"

def consultar_es(rango_tiempo):
    endpoint = f"{BASE_URL}/api/console/proxy?path=http-rest-service-*/_search&method=POST"

    payload = {
        "size": 0,
        "query": {
            "bool": {
                "must": [
                    {"match_phrase": {"request.Request.relatedParty.name": "CLARO"}},
                    {"range": {"@timestamp": {"gte": rango_tiempo, "lte": "now"}}}
                ],
                "must_not": [
                    {"match_phrase": {"request.Request.description": "signalLevelQuery"}},
                    {"match_phrase": {"request.Request.relatedParty.name": "ATP-Tecnotree"}}
                ]
            }
        },
        "aggs": {
            "conteo_por_sitio": {
                "terms": {
                    "field": CAMPO_SUCIO,
                    "size": 500
                }
            }
        }
    }

    try:
        response = requests.post(
            endpoint,
            auth=AUTH,
            headers=HEADERS,
            json=payload,
            verify=False
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error servidor: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        st.error(f"Error conexión: {e}")
        return None


# --- UI STREAMLIT ---
st.set_page_config(page_title="Dashboard Real-Time Claro", layout="wide")

st.title("📊 Monitor Real-Time: Sitios Claro")
st.caption(f"Última actualización: {datetime.now().strftime('%H:%M:%S')}")

tiempo_opcion = st.sidebar.selectbox(
    "Rango de consulta:",
    ["Últimos 15 Minutos", "Última Hora", "Últimas 24 Horas", "Últimos 7 Días"],
    index=1
)

mapping_tiempo = {
    "Últimos 15 Minutos": "now-15m",
    "Última Hora": "now-1h",
    "Últimas 24 Horas": "now-24h",
    "Últimos 7 Días": "now-7d"
}

if st.button("🔄 Sincronizar Datos Ahora"):

    res = consultar_es(mapping_tiempo[tiempo_opcion])

    if res:
        buckets = res.get("aggregations", {}).get("conteo_por_sitio", {}).get("buckets", [])

        if not buckets:
            st.warning("No hay datos en ese rango.")
        else:
            datos = []

            for b in buckets:
                valor = b["key"]
                conteo = b["doc_count"]

                partes = valor.split(",")
                nombre_sitio = partes[2].strip() if len(partes) >= 3 else "Formato Inválido"

                datos.append({
                    "Sitio": nombre_sitio,
                    "Transacciones": conteo
                })

            df = pd.DataFrame(datos)
            df_final = (
                df.groupby("Sitio")["Transacciones"]
                .sum()
                .reset_index()
                .sort_values(by="Transacciones", ascending=False)
            )

            total = df_final["Transacciones"].sum()

            st.metric("Total General", f"{total:,}")

            col1, col2 = st.columns([2,1])

            with col1:
                st.bar_chart(df_final.set_index("Sitio"))

            with col2:
                st.dataframe(df_final, use_container_width=True)

            csv = df_final.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar CSV", csv, "reporte.csv", "text/csv")