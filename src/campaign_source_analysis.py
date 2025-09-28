"""
Campaign Source Analysis - Objetivo 1

Análisis de conversión y rendimiento por fuente de campaña.

Este script realiza:
- Conexión a una base de datos SQLite preprocesada (marketing_analysis.db).
- Ejecución de queries para obtener métricas por canal y tipo de curso.
- Guardado de resultados en CSV.
- Creación de visualizaciones (barras, heatmaps y scatter plots) para análisis.
- Guardado de gráficos resultantes.

Dependencia:
- data_preprocessing.py debe ejecutarse antes para generar la base de datos.

Outputs:
- CSVs con resultados de queries.
- PNGs con visualizaciones.
- TXT con resumen ejecutivo para stakeholders.
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def setup_paths():
    """
    Configura y crea las rutas necesarias para el proyecto.

    Retorna:
        db_path (str): Ruta al archivo de base de datos SQLite.
        analysis_output_path (str): Ruta donde se guardarán los resultados del análisis.
    """
    output_path = "../output"
    processed_data_path = os.path.join(output_path, "processed_data")
    analysis_output_path = os.path.join(output_path, "campaign_source_analysis")
    db_path = os.path.join(processed_data_path, "marketing_analysis.db")
    
    # Crear carpeta si no existe
    os.makedirs(analysis_output_path, exist_ok=True)
    
    return db_path, analysis_output_path

def connect_to_database(db_path):
    """
    Conecta a la base de datos SQLite.

    Parámetros:
        db_path (str): Ruta al archivo de base de datos.

    Retorna:
        sqlite3.Connection: Objeto de conexión a la base de datos.

    Notas:
    Esta función puede lanzar FileNotFoundError si la base de datos indicada no existe.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Base de datos no encontrada: {db_path}. Ejecute data_preprocessing.py primero.")
    
    print(f"Conectando a base de datos: {db_path}")
    return sqlite3.connect(db_path)

def execute_queries(conn):
    """
    Ejecuta consultas SQL para obtener métricas clave por canal y curso.

    Parámetros:
        conn (sqlite3.Connection): Conexión activa a la base de datos.

    Retorna:
        tuple: DataFrames con métricas por canal, por curso y datos detallados.
    """
    # Consulta para métricas agregadas por canal
    query_channel = """
    SELECT
      c.channel,
      COUNT(DISTINCT l.lead_id) as total_leads,
      COUNT(DISTINCT i.inscription_id) as total_inscriptions,
      ROUND(
        (COUNT(DISTINCT i.lead_id) * 100.0 / NULLIF(COUNT(DISTINCT l.lead_id), 0)), 2
      ) as "Tasa_Conversion_Pct",
      ROUND(SUM(DISTINCT c.cost), 2) as "Costo_Total",
      ROUND(COALESCE(SUM(i.amount), 0), 2) as "Ingresos_Total",
      ROUND(
        ROUND(SUM(DISTINCT c.cost), 2) / NULLIF(COUNT(DISTINCT i.inscription_id), 0), 2
      ) as "CPA",
      ROUND(
        ((COALESCE(SUM(i.amount), 0) - ROUND(SUM(DISTINCT c.cost), 2)) * 100.0 / NULLIF(ROUND(SUM(DISTINCT c.cost), 2), 0)), 2
      ) as "ROI_Pct",
      ROUND(
        COALESCE(SUM(i.amount), 0) / NULLIF(COUNT(DISTINCT i.inscription_id), 0), 2
      ) as "Valor_Promedio_Inscripcion"
    FROM campaigns c
    LEFT JOIN leads l ON CAST(c.campaign_id AS TEXT) = CAST(l.input_channel AS TEXT)
    LEFT JOIN inscriptions i ON l.lead_id = i.lead_id
    GROUP BY c.channel
    ORDER BY "ROI_Pct" DESC;
    """

    # Consulta para métricas agregadas por tipo de curso
    query_course = """
    SELECT
      c.course_type,
      COUNT(DISTINCT l.lead_id) as total_leads,
      COUNT(DISTINCT i.inscription_id) as total_inscriptions,
      ROUND(
        (COUNT(DISTINCT i.lead_id) * 100.0 / NULLIF(COUNT(DISTINCT l.lead_id), 0)), 2
      ) as "Tasa_Conversion_Pct",
      ROUND(SUM(DISTINCT c.cost), 2) as "Costo_Total",
      ROUND(COALESCE(SUM(i.amount), 0), 2) as "Ingresos_Total",
      ROUND(
        SUM(DISTINCT c.cost) / NULLIF(COUNT(DISTINCT i.inscription_id), 0), 2
      ) as "CPA",
      ROUND(
        ((COALESCE(SUM(i.amount), 0) - SUM(DISTINCT c.cost)) * 100.0 / NULLIF(SUM(DISTINCT c.cost), 0)), 2
      ) as "ROI_Pct",
      ROUND(
        COALESCE(SUM(i.amount), 0) / NULLIF(COUNT(DISTINCT i.inscription_id), 0), 2
      ) as "Valor_Promedio_Inscripcion"
    FROM campaigns c
    LEFT JOIN leads l ON CAST(c.campaign_id AS TEXT) = CAST(l.input_channel AS TEXT)
    LEFT JOIN inscriptions i ON l.lead_id = i.lead_id
    GROUP BY c.course_type
    ORDER BY "Tasa_Conversion_Pct" DESC;
    """

    # Consulta detallada para análisis cruzado canal-curso
    query_detailed = """
    SELECT
      c.channel,
      c.course_type,
      COUNT(DISTINCT l.lead_id) as total_leads,
      COUNT(DISTINCT i.inscription_id) as total_inscriptions,
      ROUND(
        (COUNT(DISTINCT i.lead_id) * 100.0 / NULLIF(COUNT(DISTINCT l.lead_id), 0)), 2
      ) as conversion_rate,
      ROUND(SUM(DISTINCT c.cost), 2) as cost_total,
      ROUND(COALESCE(SUM(i.amount), 0), 2) as revenue_total,
      ROUND(
        ROUND(SUM(DISTINCT c.cost), 2) / NULLIF(COUNT(DISTINCT i.inscription_id), 0), 2
      ) as cpa,
      ROUND(
        ((COALESCE(SUM(i.amount), 0) - ROUND(SUM(DISTINCT c.cost), 2)) * 100.0 / NULLIF(ROUND(SUM(DISTINCT c.cost), 2), 0)), 2
      ) as roi_pct
    FROM campaigns c
    LEFT JOIN leads l ON CAST(c.campaign_id AS TEXT) = CAST(l.input_channel AS TEXT)
    LEFT JOIN inscriptions i ON l.lead_id = i.lead_id
    GROUP BY c.channel, c.course_type
    ORDER BY c.channel, c.course_type;
    """

    print("Ejecutando queries de análisis...")
    df_channel = pd.read_sql_query(query_channel, conn)
    df_course = pd.read_sql_query(query_course, conn)
    df_detailed = pd.read_sql_query(query_detailed, conn)
    
    return df_channel, df_course, df_detailed

def save_results_to_csv(df_channel, df_course, output_path):
    """
    Guarda los resultados en CSV con marca temporal.

    Parámetros:
        df_channel (pandas.DataFrame): Métricas por canal.
        df_course (pandas.DataFrame): Métricas por curso.
        output_path (str): Ruta para guardar los CSV.

    Retorna:
        str: Marca temporal usada en los nombres de archivo.
    
    Notas:
    Genera archivos CSV con sufijo de timestamp en la carpeta de salida.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    channel_file = os.path.join(output_path, f'metrics_by_channel_{timestamp}.csv')
    course_file = os.path.join(output_path, f'metrics_by_course_{timestamp}.csv')
    
    df_channel.to_csv(channel_file, index=False, encoding='utf-8')
    df_course.to_csv(course_file, index=False, encoding='utf-8')
    
    print(f"Guardado: {channel_file}")
    print(f"Guardado: {course_file}")
    
    return timestamp

def configure_plot_style():
    """
    Configura el estilo visual global para las gráficas (Matplotlib + Seaborn).
    
    Ajusta colores, tamaños de fuente y parámetros por defecto 
    para mejorar la legibilidad de los gráficos.

    Retorna: None
    """
    plt.style.use('default')
    sns.set_palette("tab10")
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 11
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 14


def create_conversion_analysis(df_channel, df_course, df_detailed, output_path, timestamp):
    """
    Crea visualizaciones gráficas del análisis de conversión.

    Parámetros:
    - df_channel (pandas.DataFrame): Datos agregados por canal.
    - df_course (pandas.DataFrame): Datos agregados por tipo de curso.
    - df_detailed (pandas.DataFrame): Datos detallados con canal, tipo de curso y tasa de conversión.
    - output_path (str): Carpeta donde guardar el gráfico.
    - timestamp (str): Sufijo para el archivo de salida, normalmente fecha/hora.
    
    Acciones:
    - Genera dos visualizaciones lado a lado:
        1. Gráfico de barras agrupadas de tasa de conversión por canal y tipo de curso.
        2. Mapa de calor de la tasa de conversión.
    - Guarda el gráfico como PNG.
    """
    print("Creando visualizaciones de análisis de conversión...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Análisis de Tasa de Conversión por Fuente de Campaña', fontsize=16, fontweight='bold')
    
    # Bar Chart Agrupado
    channels = df_detailed['channel'].unique()
    course_types = df_detailed['course_type'].unique()
    x = np.arange(len(channels))
    width = 0.8 / len(course_types)
    
    colors = sns.color_palette("tab10", len(course_types))
    
    for i, course_type in enumerate(course_types):
        data = []
        for channel in channels:
            value = df_detailed[(df_detailed['channel'] == channel) & 
                               (df_detailed['course_type'] == course_type)]['conversion_rate']
            data.append(value.iloc[0] if len(value) > 0 else 0)
        
        bars = ax1.bar(x + i * width, data, width, label=course_type, 
                       color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        for bar, val in zip(bars, data):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Canal')
    ax1.set_ylabel('Tasa de Conversión (%)')
    ax1.set_title('Tasa de Conversión por Canal y Tipo de Curso', fontweight='bold')
    ax1.set_xticks(x + width * (len(course_types) - 1) / 2)
    ax1.set_xticklabels(channels, rotation=45)
    ax1.legend(title='Tipo de Curso', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    
    # Heatmap
    pivot_data = df_detailed.pivot(index='channel', columns='course_type', values='conversion_rate')
    im = ax2.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.iloc[i, j]
            if pd.notna(value):
                text = ax2.text(j, i, f'{value:.1f}%', ha="center", va="center",
                               color="white" if value < pivot_data.values.mean() else "black",
                               fontweight='bold')
    
    ax2.set_xticks(range(len(pivot_data.columns)))
    ax2.set_yticks(range(len(pivot_data.index)))
    ax2.set_xticklabels(pivot_data.columns, rotation=45)
    ax2.set_yticklabels(pivot_data.index)
    ax2.set_xlabel('Tipo de Curso')
    ax2.set_ylabel('Canal')
    ax2.set_title('Mapa de Calor - Tasa de Conversión', fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Tasa de Conversión (%)')
    
    plt.tight_layout()
    conversion_file = os.path.join(output_path, f'conversion_analysis_{timestamp}.png')
    plt.savefig(conversion_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Guardado: {conversion_file}")

def create_performance_analysis(df_channel, output_path, timestamp):
    """
    Crea visualizaciones gráficas del análisis de rendimiento.

    Parámetros:
    - df_channel (pandas.DataFrame): Datos agregados por canal con métricas de CPA, ROI, etc.
    - output_path (str): Carpeta donde guardar el gráfico.
    - timestamp (str): Sufijo para el archivo de salida.

    Acciones:
    - Genera dos visualizaciones:
        1. Gráfico de dispersión (CPA vs ROI) con tamaño de burbuja según volumen de leads.
        2. Matriz estratégica tipo BCG.
    """
    print("Creando visualizaciones de análisis de rendimiento...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Análisis de Rendimiento por Canal', fontsize=16, fontweight='bold')
    
    # Scatter Plot 
    scatter = ax1.scatter(df_channel['CPA'], df_channel['ROI_Pct'], 
                         s=df_channel['total_leads']*2,  # Bubble size
                         c=range(len(df_channel)), 
                         cmap='viridis', 
                         alpha=0.7, 
                         edgecolors='black', 
                         linewidth=1)
    
    for i, row in df_channel.iterrows():
        ax1.annotate(row['channel'], 
                    (row['CPA'], row['ROI_Pct']),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('CPA (Costo por Adquisición €)')
    ax1.set_ylabel('ROI (%)')
    ax1.set_title('Rendimiento por Canal: CPA vs ROI\n(Tamaño burbuja = Volumen de Leads)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='ROI punto equilibrio')
    ax1.axvline(x=df_channel['CPA'].median(), color='orange', linestyle='--', alpha=0.5, label='CPA mediano')
    
    ax1.text(-0.05, 1.02, 'CPA Bajo\nROI Alto\n(ÓPTIMO)', transform=ax1.transAxes, 
             verticalalignment='bottom', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=10)
    ax1.text(1.05, 1.02, 'CPA Alto\nROI Alto\n(ESCALAR)', transform=ax1.transAxes, 
             verticalalignment='bottom', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), fontsize=10)
    ax1.text(-0.05, -0.08, 'CPA Bajo\nROI Bajo\n(OPTIMIZAR)', transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=10)
    ax1.text(1.05, -0.08, 'CPA Alto\nROI Bajo\n(ELIMINAR)', transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8), fontsize=10)
    
    ax1.legend()
    
    median_conversion = df_channel['Tasa_Conversion_Pct'].median()
    median_roi = df_channel['ROI_Pct'].median()
    
    # Matriz BCG 
    quadrants = []
    for _, row in df_channel.iterrows():
        if row['Tasa_Conversion_Pct'] >= median_conversion and row['ROI_Pct'] >= median_roi:
            quadrants.append('Estrellas\n(Alta Conv + Alto ROI)')
        elif row['Tasa_Conversion_Pct'] >= median_conversion and row['ROI_Pct'] < median_roi:
            quadrants.append('Interrogantes\n(Alta Conv + Bajo ROI)')
        elif row['Tasa_Conversion_Pct'] < median_conversion and row['ROI_Pct'] >= median_roi:
            quadrants.append('Vacas Lecheras\n(Baja Conv + Alto ROI)')
        else:
            quadrants.append('Perros\n(Baja Conv + Bajo ROI)')
    
    df_channel['quadrant'] = quadrants
    
    for i, (quad, color) in enumerate([
        ('Estrellas\n(Alta Conv + Alto ROI)', 'green'),
        ('Interrogantes\n(Alta Conv + Bajo ROI)', 'orange'), 
        ('Vacas Lecheras\n(Baja Conv + Alto ROI)', 'blue'),
        ('Perros\n(Baja Conv + Bajo ROI)', 'red')
    ]):
        quad_data = df_channel[df_channel['quadrant'] == quad]
        if not quad_data.empty:
            ax2.scatter(quad_data['Tasa_Conversion_Pct'], quad_data['ROI_Pct'],
                       s=quad_data['total_leads']*3, 
                       c=color, alpha=0.7, label=quad.split('\n')[0],
                       edgecolors='black', linewidth=1)
            
            for _, row in quad_data.iterrows():
                ax2.annotate(row['channel'], 
                            (row['Tasa_Conversion_Pct'], row['ROI_Pct']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax2.axhline(y=median_roi, color='black', linestyle='--', alpha=0.5, label=f'ROI Mediano ({median_roi:.1f}%)')
    ax2.axvline(x=median_conversion, color='black', linestyle='--', alpha=0.5, label=f'Conversión Mediana ({median_conversion:.1f}%)')
    ax2.set_xlabel('Tasa de Conversión (%)')
    ax2.set_ylabel('ROI (%)')
    ax2.set_title('Matriz Estratégica de Rendimiento\n(Tamaño burbuja = Volumen de Leads)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    performance_file = os.path.join(output_path, f'performance_analysis_{timestamp}.png')
    plt.savefig(performance_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Guardado: {performance_file}")

def generate_executive_summary(df_channel, df_course, output_path, timestamp):
    """
    Genera un resumen ejecutivo del rendimiento de las campañas y lo guarda como archivo .txt.

    Parámetros:
    df_channel : pandas.DataFrame
        DataFrame con métricas por canal de adquisición.
        Debe contener columnas: 'total_leads', 'total_inscriptions', 'Costo_Total', 
        'Ingresos_Total', 'Tasa_Conversion_Pct', 'ROI_Pct', 'channel'.
        
    df_course : pandas.DataFrame
        DataFrame con métricas por tipo de curso.
        Debe contener columnas: 'Tasa_Conversion_Pct', 'course_type'.
        
    output_path : str
        Ruta del directorio donde se guardará el archivo de resumen.
        
    timestamp : str
        Sufijo temporal para nombrar el archivo de salida.
        """
    print("Generando resumen ejecutivo...")
    
    # Calcular métricas globales
    total_leads = df_channel['total_leads'].sum()
    total_inscriptions = df_channel['total_inscriptions'].sum()
    total_cost = df_channel['Costo_Total'].sum()
    total_revenue = df_channel['Ingresos_Total'].sum()
    overall_conversion = (total_inscriptions / total_leads * 100) if total_leads > 0 else 0
    overall_roi = ((total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
    overall_cpa = (total_cost / total_inscriptions) if total_inscriptions > 0 else 0
    
    # Mejores performers
    best_channel_conversion = df_channel.loc[df_channel['Tasa_Conversion_Pct'].idxmax()]
    best_course_conversion = df_course.loc[df_course['Tasa_Conversion_Pct'].idxmax()]
    best_roi_channel = df_channel.loc[df_channel['ROI_Pct'].idxmax()]
    
    report = f"""
RESUMEN EJECUTIVO DE CAMPAÑAS - OBJETIVO 1
Fecha de Análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MÉTRICAS GLOBALES:
• Total Leads: {total_leads:,}
• Total Inscripciones: {total_inscriptions:,}
• Tasa de Conversión Global: {overall_conversion:.2f}%
• ROI Global: {overall_roi:.2f}%
• CPA Global: €{overall_cpa:.2f}
• Inversión Total: €{total_cost:,.2f}
• Ingresos Totales: €{total_revenue:,.2f}

MEJORES PERFORMERS:
• Mejor Canal (Conversión): {best_channel_conversion['channel']} ({best_channel_conversion['Tasa_Conversion_Pct']:.2f}%)
• Mejor Curso (Conversión): {best_course_conversion['course_type']} ({best_course_conversion['Tasa_Conversion_Pct']:.2f}%)
• Mejor Canal (ROI): {best_roi_channel['channel']} ({best_roi_channel['ROI_Pct']:.2f}%)

RECOMENDACIONES ESTRATÉGICAS:
• Enfocar presupuesto en: {best_roi_channel['channel']}
• Promover más: {best_course_conversion['course_type']}
• Revisar canales con ROI negativo para optimización
"""
    
    # Guardar report
    report_file = os.path.join(output_path, f'executive_summary_{timestamp}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"Guardado: {report_file}")

def display_results_summary(df_channel, df_course):
    """
    Muestra en consola un resumen de resultados por canal y por tipo de curso.

    Parámetros:
    df_channel : pandas.DataFrame
        DataFrame con métricas por canal de adquisición.

    df_course : pandas.DataFrame
        DataFrame con métricas por tipo de curso.
    """
    print("RESULTADOS POR CANAL:")
    print(df_channel.to_string(index=False))
    
    print("RESULTADOS POR TIPO DE CURSO:")
    print(df_course.to_string(index=False))

def main():
    """
    Función principal del análisis de fuentes de campaña - Objetivo 1.
    Ejecuta el flujo completo:
    1. Configuración de rutas.
    2. Conexión a base de datos.
    3. Ejecución de queries.
    4. Guardado de resultados en CSV.
    5. Muestra de resumen de resultados.
    6. Configuración de estilo gráfico.
    7. Creación de visualizaciones.
    8. Generación de resumen ejecutivo.
    9. Cierre de conexión.
    """
    print("Iniciando análisis de fuentes de campaña - Objetivo 1...")
    
    # 1. Configurar rutas
    db_path, output_path = setup_paths()
    
    # 2. Conectar a base de datos
    conn = connect_to_database(db_path)
    
    # 3. Ejecutar queries
    df_channel, df_course, df_detailed = execute_queries(conn)
    
    # 4. Guardar resultados en CSV
    timestamp = save_results_to_csv(df_channel, df_course, output_path)
    
    # 5. Mostrar resultados
    display_results_summary(df_channel, df_course)
    
    # 6. Configurar estilo de visualizaciones
    configure_plot_style()
    
    # 7. Crear visualizaciones de conversión
    create_conversion_analysis(df_channel, df_course, df_detailed, output_path, timestamp)
    
    # 8. Crear visualizaciones de rendimiento
    create_performance_analysis(df_channel, output_path, timestamp)
    
    # 9. Generar resumen ejecutivo
    generate_executive_summary(df_channel, df_course, output_path, timestamp)
    
    # 10. Cerrar conexión
    conn.close()
    
    print("ANÁLISIS DE FUENTES DE CAMPAÑA COMPLETADO!")
    print(f"Todos los archivos guardados en: {output_path}")


if __name__ == "__main__":
    main()