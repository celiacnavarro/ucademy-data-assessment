"""
Script de Preprocesamiento de Datos de Marketing

Este script limpia y prepara datos de marketing (campañas, inscripciones y leads) para su análisis posterior.
Genera:
- Datos limpios.
- Estadísticas resumen.
- Una base de datos SQLite reutilizable.
- Archivos CSV de respaldo.
- Un reporte en formato JSON y TXT con estadísticas.

Estructura:
1. Configuración de rutas.
2. Carga de datos originales.
3. Limpieza y transformación.
4. Categorización de fuentes de leads.
5. Creación de estadísticas resumen.
6. Guardado en base de datos y archivos.

Outputs:
- CSVs con los datos preprocesados. 
- Base de datos 'marketing_analysis.db'
- JSON y TXT con resumen exploratorio de los datos.
"""

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime

# 1. Configuración de rutas

def setup_paths():
    """
    Configura las rutas necesarias para el procesamiento.

    Retorna:
        data_path (str): Ruta donde están los datos originales.
        processed_data_path (str): Ruta donde se guardarán datos procesados.
        db_path (str): Ruta del archivo SQLite que contendrá los datos limpios.
    """
    data_path = "../data"
    output_path = "../output"
    processed_data_path = os.path.join(output_path, "processed_data")
    db_path = os.path.join(processed_data_path, "marketing_analysis.db")
    
    # Crear directorio si no existe
    os.makedirs(processed_data_path, exist_ok=True)
    
    return data_path, processed_data_path, db_path


# 2. Carga de datos originales

def load_raw_data(data_path):
    """
    Carga los datos JSON originales desde la ruta especificada.

    Parámetros:
        data_path (str): Ruta donde se encuentran los archivos JSON.

    Retorna:
        campaigns (pandas.DataFrame): Datos de campañas.
        inscriptions (pandas.DataFrame): Datos de inscripciones.
        leads (pandas.DataFrame): Datos de leads.
    """
    print("Cargando archivos de datos originales...")

    campaigns = pd.read_json(os.path.join(data_path, "campaigns.json"))
    inscriptions = pd.read_json(os.path.join(data_path, "inscriptions.json"))
    leads = pd.read_json(os.path.join(data_path, "leads.json"))

    print(f"Cargado: {len(campaigns)} campañas, {len(leads)} leads, {len(inscriptions)} inscripciones")
    return campaigns, inscriptions, leads


# 3. Limpieza de datos financieros

def clean_financial_data(df, column):
    """
    Limpia columnas financieras eliminando símbolos (€) y convirtiendo comas a puntos.
    Convierte el resultado a tipo float.

    Parámetros:
        df (pandas.DataFrame): DataFrame que contiene la columna.
        column (str): Nombre de la columna a limpiar.

    Retorna:
        Series (float): Columna limpia y convertida a float.
    """
    return (df[column]
            .str.replace("€", "", regex=False)
            .str.replace(",", ".", regex=False)
            .astype(float))


# 4. Limpieza específica de cada tabla

def clean_campaigns(campaigns):
    """
    Limpia y transforma la tabla de campañas.

    Transformaciones:
    - Limpia costos financieros.
    - Normaliza texto.
    - Extrae tipo de curso y canal desde el nombre.
    - Convierte fechas a datetime.

    Parámetros:
        campaigns (pandas.DataFrame): Tabla original de campañas.

    Retorna:
        campaigns (pandas.DataFrame): Tabla limpia y transformada.
    """
    print("Limpiando datos de campañas...")

    campaigns['cost'] = clean_financial_data(campaigns, 'cost')
    campaigns["name"] = campaigns["name"].str.lower().str.strip()
    campaigns["campaign_id"] = campaigns["campaign_id"].str.lower().str.strip()

    # Extraer course_type y channel
    campaigns["course_type"] = campaigns["name"].apply(
        lambda x: x.split("_")[0] if "_" in x else "unknown"
    )
    campaigns["channel"] = campaigns["name"].apply(
        lambda x: x.split("_")[1] if "_" in x else "unknown"
    )

    # Convertir fechas
    campaigns['started_at'] = pd.to_datetime(campaigns['started_at'])
    campaigns['ended_at'] = pd.to_datetime(campaigns['ended_at'])

    print(f"Procesadas {len(campaigns)} campañas")
    return campaigns


def clean_inscriptions(inscriptions):
    """
    Limpia y transforma la tabla de inscripciones.

    Transformaciones:
    - Limpia montos financieros.
    - Convierte fechas a datetime.

    Parámetros:
        inscriptions (pandas.DataFrame): Tabla original de inscripciones.

    Retorna:
        inscriptions (pandas.DataFrame): Tabla limpia y transformada.
    """
    print("Limpiando datos de inscripciones...")
    inscriptions['amount'] = clean_financial_data(inscriptions, 'amount')
    inscriptions['created_at'] = pd.to_datetime(inscriptions['created_at'])
    print(f"Procesadas {len(inscriptions)} inscripciones")
    return inscriptions


def clean_leads(leads):
    """
    Limpia y transforma la tabla de leads.

    Transformaciones:
    - Normaliza canales de entrada.
    - Convierte fechas a datetime.

    Parámetros:
        leads (pandas.DataFrame): Tabla original de leads.

    Retorna:
        leads (pandas.DataFrame): Tabla limpia y transformada.
    """
    print("Limpiando datos de leads...")
    leads["input_channel"] = leads["input_channel"].str.lower().str.strip()
    leads['created_at'] = pd.to_datetime(leads['created_at'])
    print(f"Procesados {len(leads)} leads")
    return leads


# 5. Categorización de leads

def categorize_all_lead_sources(leads, campaigns):
    """
    Categorización de leads según su fuente.

    Clasificación:
    - paid_campaign: proveniente de una campaña pagada.
    - organic: proveniente de búsqueda orgánica.
    - partner: proveniente de partners/referrals.
    - other: cualquier otro caso.

    Parámetros:
        leads (pandas.DataFrame): Tabla de leads.
        campaigns (pandas.DataFrame): Tabla de campañas limpias.

    Retorna:
        leads (pandas.DataFrame): Leads con columnas adicionales de categorización.
    """
    print("Categorizando fuentes de leads...")

    campaign_mapping = campaigns.set_index('campaign_id')[['channel', 'course_type']].to_dict('index')

    def categorize_source(input_channel):
        if pd.isna(input_channel) or input_channel == '' or input_channel == 'null':
            return 'unknown', 'unknown', 'unknown'
        input_channel_lower = str(input_channel).lower().strip()

        if input_channel_lower in campaign_mapping:
            campaign_info = campaign_mapping[input_channel_lower]
            return 'paid_campaign', campaign_info['channel'], campaign_info['course_type']
        elif 'organic' in input_channel_lower or 'web_organic' in input_channel_lower:
            return 'organic', 'web', 'unknown'
        elif 'partner' in input_channel_lower or 'ref' in input_channel_lower:
            return 'partner', 'referral', 'unknown'
        else:
            return 'other', 'unknown', 'unknown'

    categorization_results = leads['input_channel'].apply(categorize_source)
    leads['source_category'] = [result[0] for result in categorization_results]
    leads['channel_derived'] = [result[1] for result in categorization_results]
    leads['course_type_derived'] = [result[2] for result in categorization_results]

    print("Distribución de categorías de fuentes:")
    print(leads['source_category'].value_counts())

    return leads


# 6. Estadísticas resumen

def create_summary_stats(campaigns, leads, inscriptions):
    """
    Genera estadísticas generales de los datos procesados.

    Parámetros:
        campaigns (pandas.DataFrame): Campañas limpias.
        leads (pandas.DataFrame): Leads limpios.
        inscriptions (pandas.DataFrame): Inscripciones limpias.

    Retorna:
        stats (dict): Estadísticas resumidas.
    """
    print("Generando estadísticas resumen...")
    stats = {
        'data_overview': {
            'total_campaigns': len(campaigns),
            'total_leads': len(leads),
            'total_inscriptions': len(inscriptions),
            'date_range_leads': f"{leads['created_at'].min()} to {leads['created_at'].max()}",
            'date_range_inscriptions': f"{inscriptions['created_at'].min()} to {inscriptions['created_at'].max()}",
        },
        'lead_sources': leads['source_category'].value_counts().to_dict(),
        'campaign_channels': campaigns['channel'].value_counts().to_dict(),
        'course_types': campaigns['course_type'].value_counts().to_dict(),
        'processing_date': datetime.now().isoformat()
    }
    return stats


# 7. Guardado de datos

def save_to_database(campaigns, leads, inscriptions, db_path):
    """
    Guarda los DataFrames limpios en una base de datos SQLite.

    Este procedimiento persiste las tablas procesadas en un fichero SQLite para
    facilitar consultas posteriores desde los scripts de análisis. Las tablas se
    sobrescriben en cada ejecución (if_exists="replace").

    Parámetros:
        campaigns (pandas.DataFrame): DataFrame con la tabla `campaigns` ya limpiada.
        leads (pandas.DataFrame): DataFrame con la tabla `leads` ya limpiada.
        inscriptions (pandas.DataFrame): DataFrame con la tabla `inscriptions` ya limpiada.
        db_path (str): Ruta completa al fichero de base de datos SQLite donde se guardarán las tablas.
    
    Retorna:
        None
    
    Notas:
    Sobrescribe las tablas 'campaigns', 'leads' e 'inscriptions' en la base de datos SQLite.

    """
    print("Guardando datos limpios en base de datos SQLite...")
    conn = sqlite3.connect(db_path)
    campaigns.to_sql("campaigns", conn, if_exists="replace", index=False)
    inscriptions.to_sql("inscriptions", conn, if_exists="replace", index=False)
    leads.to_sql("leads", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Base de datos guardada: {db_path}")


def save_processed_csvs(campaigns, leads, inscriptions, processed_data_path):
    """
    Guarda copias de seguridad en CSV de los datos procesados.

    Parámetros:
        campaigns (pandas.DataFrame): Datos de campañas procesados.
        leads (pandas.DataFrame): Datos de leads procesados.
        inscriptions (pandas.DataFrame): Datos de inscripciones procesados.
        processed_data_path (str): Carpeta donde se guardarán los CSVs.

    Retorna:
        None
    
    Notas:
    Genera 3 archivos CSV con timestamp en el directorio de salida.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaigns.to_csv(os.path.join(processed_data_path, f'campaigns_clean_{timestamp}.csv'), index=False)
    leads.to_csv(os.path.join(processed_data_path, f'leads_clean_{timestamp}.csv'), index=False)
    inscriptions.to_csv(os.path.join(processed_data_path, f'inscriptions_clean_{timestamp}.csv'), index=False)
    print(f"CSVs de backup guardados con timestamp: {timestamp}")


def save_summary_report(stats, processed_data_path):
    """
    Guarda un reporte resumen del procesamiento de datos.

    Parámetros:
    stats : dict
        Diccionario con estadísticas y métricas generadas durante el procesamiento.
        Debe contener al menos:
            - 'processing_date': str, fecha del procesamiento.
            - 'data_overview': dict con totales y rangos de fechas.
            - 'lead_sources': dict con conteos por fuente de lead.
            - 'campaign_channels': dict con conteos por canal.
            - 'course_types': dict con conteos por tipo de curso.
    processed_data_path : str
        Ruta donde se guardarán los reportes generados.

    Retorna:
    report : str
        Texto del reporte generado, útil para imprimir en consola.
    
    Notas:
    Genera dos archivos en processed_data_path:
    - data_summary_<timestamp>.json
    - data_summary_<timestamp>.txt

    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Guardar JSON de estadísticas
    with open(os.path.join(processed_data_path, f'data_summary_{timestamp}.json'), 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    # Generar reporte TXT legible
    report = f"""
RESUMEN DE PROCESAMIENTO DE DATOS
Fecha de Procesamiento: {stats['processing_date']}

RESUMEN DE DATOS:
• Total Campañas: {stats['data_overview']['total_campaigns']:,}
• Total Leads: {stats['data_overview']['total_leads']:,}
• Total Inscripciones: {stats['data_overview']['total_inscriptions']:,}
• Rango Fechas Leads: {stats['data_overview']['date_range_leads']}
• Rango Fechas Inscripciones: {stats['data_overview']['date_range_inscriptions']}

BREAKDOWN POR FUENTES DE LEADS:
"""
    for source, count in stats['lead_sources'].items():
        percentage = (count / stats['data_overview']['total_leads']) * 100
        report += f"• {source.title()}: {count:,} ({percentage:.1f}%)\n"

    report += f"\nCANALES DE CAMPAÑA:\n"
    for channel, count in stats['campaign_channels'].items():
        report += f"• {channel.title()}: {count} campañas\n"

    report += f"\nTIPOS DE CURSO:\n"
    for course, count in stats['course_types'].items():
        report += f"• {course.title()}: {count} campañas\n"

    # Guardar reporte TXT
    with open(os.path.join(processed_data_path, f'data_summary_{timestamp}.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Reporte resumen guardado en {processed_data_path}")
    return report


def main():
    """
    Función principal de preprocesamiento de datos.

    Flujo de trabajo:
    1. Configura rutas de datos.
    2. Carga datasets originales: campañas, inscripciones, leads.
    3. Limpia los datos cargados.
    4. Categoriza leads según canal de origen.
    5. Crea estadísticas resumen.
    6. Guarda datos en base de datos.
    7. Guarda copias procesadas en CSV.
    8. Genera y guarda reporte resumen.

    Retorna:
    db_path : str
        Ruta donde se guardó la base de datos procesada.
    processed_data_path : str
        Ruta donde se guardaron los datos procesados y reportes.
    """
    print("Iniciando pipeline de preprocesamiento de datos...")

    # 1. Configurar rutas
    data_path, processed_data_path, db_path = setup_paths()

    # 2. Cargar datos originales
    campaigns, inscriptions, leads = load_raw_data(data_path)

    # 3. Limpiar datos
    campaigns_clean = clean_campaigns(campaigns)
    inscriptions_clean = clean_inscriptions(inscriptions)
    leads_clean = clean_leads(leads)

    # 4. Categorizar leads
    leads_clean = categorize_all_lead_sources(leads_clean, campaigns_clean)

    # 5. Crear estadísticas
    stats = create_summary_stats(campaigns_clean, leads_clean, inscriptions_clean)

    # 6. Guardar en base de datos
    save_to_database(campaigns_clean, leads_clean, inscriptions_clean, db_path)

    # 7. Guardar CSVs de backup
    save_processed_csvs(campaigns_clean, leads_clean, inscriptions_clean, processed_data_path)

    # 8. Guardar reporte resumen
    report = save_summary_report(stats, processed_data_path)

    print("PREPROCESAMIENTO DE DATOS COMPLETADO!")
    print(report)

    return db_path, processed_data_path


if __name__ == "__main__":
    db_path, processed_data_path = main()

