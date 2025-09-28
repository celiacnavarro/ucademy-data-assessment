"""
Lead Conversion Funnel Analysis - Objetivo 2

Este script realiza un análisis detallado del embudo de conversión de leads a inscripciones,
utilizando los datos preprocesados por `data_preprocessing.py`.

Funcionalidades principales:
- Carga de datos desde SQLite (base generada por el preprocesamiento).
- Análisis del embudo de conversión por fuente de leads y campañas.
- Análisis temporal (evolución semanal de leads, inscripciones y conversión).
- Análisis de calidad de leads y eficiencia de campañas.
- Visualizaciones interactivas (Plotly) y estáticas (Matplotlib/Seaborn).
- Generación de outputs en formato CSV, PNG y TXT (resumen ejecutivo).

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
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from PIL import Image


def setup_paths():
    """
    Configura las rutas de entrada y salida del proyecto.

    Retorna:
        tuple: 
            - db_path (str): Ruta a la base de datos SQLite procesada.
            - analysis_output_path (str): Carpeta donde se guardarán los resultados.
    """
    output_path = "../output"
    processed_data_path = os.path.join(output_path, "processed_data")
    analysis_output_path = os.path.join(output_path, "lead_conversion_funnel")
    db_path = os.path.join(processed_data_path, "marketing_analysis.db")
    
    os.makedirs(analysis_output_path, exist_ok=True)
    
    return db_path, analysis_output_path

def connect_to_database(db_path):
    """
    Conecta a la base de datos SQLite.

    Parámetros:
        db_path (str): Ruta de la base de datos SQLite.

    Retorna:
        sqlite3.Connection: Objeto de conexión.
    
    Notas:
    Esta función puede lanzar FileNotFoundError si la base de datos indicada no existe.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Base de datos no encontrada: {db_path}. Ejecute data_preprocessing.py primero.")
    
    print(f"Conectando a base de datos: {db_path}")
    return sqlite3.connect(db_path)

def execute_funnel_queries(conn):
    """
    Ejecuta queries SQL para obtener diferentes perspectivas del embudo:
    - Conversión por fuente de leads.
    - Rendimiento por campaña.
    - Evolución temporal semanal.
    - Calidad de leads.

    Parámetros:
        conn (sqlite3.Connection): Conexión a la base de datos.

    Retorna:
        tuple: 
            - df_funnel (DataFrame)
            - df_campaigns (DataFrame)
            - df_temporal (DataFrame)
            - df_quality (DataFrame)
    """
    
    # Query principal del embudo por fuente
    funnel_query = """
    SELECT
      CASE 
        WHEN l.source_category = 'paid_campaign' THEN 'Campañas Pagadas'
        WHEN l.source_category = 'organic' THEN 'Tráfico Orgánico'
        WHEN l.source_category = 'partner' THEN 'Partners/Referidos'
        ELSE 'Otros/Desconocido'
      END as source_group,
      l.source_category,
      COUNT(DISTINCT l.lead_id) as total_leads,
      COUNT(DISTINCT i.inscription_id) as total_inscriptions,
      ROUND(
        (COUNT(DISTINCT i.lead_id) * 100.0 / NULLIF(COUNT(DISTINCT l.lead_id), 0)), 2
      ) as conversion_rate,
      ROUND(COALESCE(AVG(i.amount), 0), 2) as avg_revenue_per_inscription
    FROM leads l
    LEFT JOIN inscriptions i ON l.lead_id = i.lead_id
    GROUP BY l.source_category
    ORDER BY total_leads DESC;
    """
    
    # Query por campaña específica (solo campañas pagadas)
    campaign_query = """
    SELECT
      c.name as campaign_name,
      c.channel,
      c.course_type,
      COUNT(DISTINCT l.lead_id) as total_leads,
      COUNT(DISTINCT i.inscription_id) as total_inscriptions,
      ROUND(
        (COUNT(DISTINCT i.lead_id) * 100.0 / NULLIF(COUNT(DISTINCT l.lead_id), 0)), 2
      ) as conversion_rate,
      ROUND(COALESCE(SUM(i.amount), 0), 2) as total_revenue,
      c.cost as campaign_cost
    FROM campaigns c
    LEFT JOIN leads l ON CAST(c.campaign_id AS TEXT) = CAST(l.input_channel AS TEXT)
    LEFT JOIN inscriptions i ON l.lead_id = i.lead_id
    GROUP BY c.campaign_id, c.name, c.channel, c.course_type, c.cost
    HAVING total_leads > 0
    ORDER BY conversion_rate DESC;
    """
    
    # Query temporal por semana
    temporal_query = """
    SELECT
      DATE(l.created_at, 'weekday 0', '-6 days') as week_start,
      CASE 
        WHEN l.source_category = 'paid_campaign' THEN 'Campañas Pagadas'
        WHEN l.source_category = 'organic' THEN 'Tráfico Orgánico'
        WHEN l.source_category = 'partner' THEN 'Partners/Referidos'
        ELSE 'Otros/Desconocido'
      END as source_group,
      COUNT(DISTINCT l.lead_id) as weekly_leads,
      COUNT(DISTINCT i.inscription_id) as weekly_inscriptions,
      ROUND(
        (COUNT(DISTINCT i.lead_id) * 100.0 / NULLIF(COUNT(DISTINCT l.lead_id), 0)), 2
      ) as weekly_conversion_rate
    FROM leads l
    LEFT JOIN inscriptions i ON l.lead_id = i.lead_id
    WHERE l.created_at IS NOT NULL
    GROUP BY week_start, l.source_category
    ORDER BY week_start, l.source_category;
    """
    
    # Query detallado para análisis de calidad de leads
    quality_query = """
    SELECT
      l.source_category,
      CASE 
        WHEN l.source_category = 'paid_campaign' THEN c.channel
        WHEN l.source_category = 'organic' THEN 'web'
        WHEN l.source_category = 'partner' THEN 'referral'
        ELSE 'unknown'
      END as channel,
      CASE 
        WHEN l.source_category = 'paid_campaign' THEN c.course_type
        ELSE 'mixed'
      END as course_type,
      COUNT(DISTINCT l.lead_id) as leads_count,
      COUNT(DISTINCT i.inscription_id) as inscriptions_count,
      ROUND(AVG(JULIANDAY(i.created_at) - JULIANDAY(l.created_at)), 1) as avg_days_to_convert,
      ROUND(COALESCE(AVG(i.amount), 0), 2) as avg_inscription_value
    FROM leads l
    LEFT JOIN inscriptions i ON l.lead_id = i.lead_id
    LEFT JOIN campaigns c ON CAST(c.campaign_id AS TEXT) = CAST(l.input_channel AS TEXT)
    GROUP BY l.source_category, channel, course_type
    HAVING leads_count >= 5
    ORDER BY l.source_category, leads_count DESC;
    """
    
    print("Ejecutando queries de análisis de embudo...")
    df_funnel = pd.read_sql_query(funnel_query, conn)
    df_campaigns = pd.read_sql_query(campaign_query, conn)
    df_temporal = pd.read_sql_query(temporal_query, conn)
    df_quality = pd.read_sql_query(quality_query, conn)
    
    return df_funnel, df_campaigns, df_temporal, df_quality

def save_results_to_csv(df_funnel, df_campaigns, df_temporal, df_quality, output_path):
    """
    Guarda los resultados de los análisis en archivos CSV con un timestamp único.

    Parámetros:
        df_funnel (pd.DataFrame): Resultados del análisis del embudo.
        df_campaigns (pd.DataFrame): Resultados del análisis por campañas.
        df_temporal (pd.DataFrame): Resultados del análisis temporal (evolución semanal).
        df_quality (pd.DataFrame): Resultados del análisis de calidad de leads.
        output_path (str): Carpeta donde se guardarán los archivos.

    Retorna:
        str: Timestamp utilizado para nombrar los archivos exportados.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    files = {
        'funnel_analysis': df_funnel,
        'campaign_conversion': df_campaigns,
        'temporal_analysis': df_temporal,
        'lead_quality_analysis': df_quality
    }
    
    saved_files = []
    for name, df in files.items():
        file_path = os.path.join(output_path, f'{name}_{timestamp}.csv')
        df.to_csv(file_path, index=False, encoding='utf-8')
        saved_files.append(file_path)
        print(f"Guardado: {file_path}")
    
    return timestamp

def configure_plot_style():
    """
    Configura el estilo visual global para las gráficas (Matplotlib + Seaborn).
    
    Ajusta colores, tamaños de fuente y parámetros por defecto 
    para mejorar la legibilidad de los gráficos.
    """
    plt.style.use('default')
    sns.set_palette("tab10")
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 11
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 14

def create_funnel_visualization(df_funnel, output_path, timestamp):
    """
    Genera visualizaciones del embudo de conversión a partir del DataFrame `df_funnel`.

    - Embudos individuales por fuente de leads (Plotly).
    - Comparativa de volumen y tasa de conversión (Matplotlib).

    Parámetros:
        df_funnel (pd.DataFrame): Resultados de conversión por fuente de leads.
        output_path (str): Carpeta donde guardar las imágenes.
        timestamp (str): Identificador único de la ejecución.

    Retorna:
        - Imagen PNG con embudos por fuente (funnel_by_source).
        - Imagen PNG con comparativa (funnel_comparison).
    """
    print("Creando visualización de embudo de conversión...")

    # Embudos por Fuente (subplots)
    sources = df_funnel['source_group'].tolist()
    n_sources = len(sources)
    
    # Crear subplots
    fig_sources = make_subplots(
        rows=1, cols=n_sources,
        specs=[[{"type": "funnel"}] * n_sources],
        subplot_titles=[f"{source}<br>{row['conversion_rate']:.1f}% conversión" 
                       for source, (_, row) in zip(sources, df_funnel.iterrows())],
        horizontal_spacing=0.1
    )
    
    tab10 = plt.get_cmap("tab10").colors
    colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in tab10]
    
    for i, (_, row) in enumerate(df_funnel.iterrows()):
        # Datos del embudo para esta fuente
        source_data = {
            'stage': ['Leads', 'Inscripciones'],
            'count': [row['total_leads'], row['total_inscriptions']],
            'percentage': [100, row['conversion_rate']]
        }
                
        # Personalizar texto: solo número en Leads, número + % en Inscripciones
        custom_text = [
            f"{source_data['count'][0]:,}",  # Leads → solo número
            f"{source_data['count'][1]:,}<br>({source_data['percentage'][1]:.1f}%)"  # Inscripciones → número + %
            ]
        
        fig_sources.add_trace(
            go.Funnel(
                y=source_data['stage'],
                x=source_data['count'],
                text=custom_text,
                textinfo="text",
                textposition="auto",
                insidetextfont=dict(size=16, color="white"),
                outsidetextfont=dict(size=16, color="black"),
                marker=dict(color=[colors[i % len(colors)], colors[i % len(colors)]]),
                opacity=0.8,
                name=row['source_group']
            ),
            row=1, col=i+1
        )

        # Ocultar labels en todos excepto el primero
        if i > 0:
            fig_sources.update_yaxes(showticklabels=False, row=1, col=i+1)

    fig_sources.update_layout(
        title=dict(
            text="Embudo de Conversión por Fuente de Leads",
            font=dict(size=18, family="Arial"),
            x=0.5
        ),
        showlegend=False,
        height=500,
        font=dict(size=16, family="Arial")
    )
    
    # Guardar embudos por fuente
    sources_png = os.path.join(output_path, f'funnel_by_source_{timestamp}.png')
    fig_sources.write_image(sources_png, width=1200, height=500, scale=2)
    img = Image.open(sources_png)
    img.show()

    # 3. Crear también una comparativa en matplotlib para complementar
    fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig_comp.suptitle('Análisis Comparativo del Embudo de Conversión', fontsize=16, fontweight='bold')
    
    # Gráfico de barras con leads e inscripciones lado a lado
    x = np.arange(len(df_funnel))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df_funnel['total_leads'], width, 
                    label='Leads', color='blue', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, df_funnel['total_inscriptions'], width,
                    label='Inscripciones', color='orange', alpha=0.8, edgecolor='black')
    
    # Añadir valores en las barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Fuente de Leads')
    ax1.set_ylabel('Cantidad')
    ax1.set_title('Volumen: Leads vs Inscripciones por Fuente', fontweight='bold')
    ax1.set_xticks(x, labels=df_funnel['source_group'])
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Gráfico de tasas de conversión
    bars3 = ax2.bar(df_funnel['source_group'], df_funnel['conversion_rate'], 
                    color=['green' if rate >= df_funnel['conversion_rate'].mean() 
                           else 'red' for rate in df_funnel['conversion_rate']], 
                    alpha=0.8, edgecolor='black')
    
    # Añadir valores en las barras
    for bar, rate in zip(bars3, df_funnel['conversion_rate']):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Línea de promedio
    avg_conversion = df_funnel['conversion_rate'].mean()
    ax2.axhline(y=avg_conversion, color='red', linestyle='--', alpha=0.7, 
               label=f'Promedio: {avg_conversion:.1f}%', linewidth=2)
    
    ax2.set_xlabel('Fuente de Leads')
    ax2.set_ylabel('Tasa de Conversión (%)')
    ax2.set_title('Eficiencia: Tasa de Conversión por Fuente', fontweight='bold')
    ax2.set_xticks(x, labels=df_funnel['source_group'])
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    comparison_file = os.path.join(output_path, f'funnel_comparison_{timestamp}.png')
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Guardado comparativa: {comparison_file}")
    
def create_temporal_analysis(df_temporal, output_path, timestamp):
    """
    Crear análisis temporal de leads e inscripciones semanales.

    Parámetros:
    df_temporal : pd.DataFrame
        DataFrame con métricas semanales por fuente:
        - week_start
        - source_group
        - weekly_leads
        - weekly_inscriptions
        - weekly_conversion_rate
    output_path : str
        Carpeta donde guardar los resultados.
    timestamp : str
        Marca de tiempo usada en el nombre del archivo de salida.

    Retorna:
    - Gráfico PNG con 4 paneles:
        1. Evolución de leads por fuente.
        2. Evolución de inscripciones por fuente.
        3. Evolución de la tasa de conversión por fuente.
        4. Eficiencia global del embudo (inscripciones/leads).
    """
    print("Creando análisis temporal...")
    
    # Preparar datos temporales
    df_temporal['week_start'] = pd.to_datetime(df_temporal['week_start'])
    df_temporal_pivot = df_temporal.pivot(index='week_start', 
                                         columns='source_group', 
                                         values=['weekly_leads', 'weekly_inscriptions', 'weekly_conversion_rate']).fillna(0)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('Evolución Temporal del Embudo de Conversión', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, 4))
    
    # 1. Evolución de Leads por fuente
    for i, source in enumerate(df_temporal_pivot['weekly_leads'].columns):
        if source in df_temporal_pivot['weekly_leads'].columns:
            ax1.plot(df_temporal_pivot.index, df_temporal_pivot['weekly_leads'][source], 
                    marker=None, linewidth=2.5, markersize=6, label=source, color=colors[i])
    
    ax1.set_title('Evolución Semanal de Leads por Fuente', fontweight='bold')
    ax1.set_xlabel('Semana')
    ax1.set_ylabel('Número de Leads')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Evolución de Inscripciones por fuente
    for i, source in enumerate(df_temporal_pivot['weekly_inscriptions'].columns):
        if source in df_temporal_pivot['weekly_inscriptions'].columns:
            ax2.plot(df_temporal_pivot.index, df_temporal_pivot['weekly_inscriptions'][source], 
                    marker=None, linewidth=2.5, markersize=6, label=source, color=colors[i])
    
    ax2.set_title('Evolución Semanal de Inscripciones por Fuente', fontweight='bold')
    ax2.set_xlabel('Semana')
    ax2.set_ylabel('Número de Inscripciones')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Tasas de conversión temporales
    for i, source in enumerate(df_temporal_pivot['weekly_conversion_rate'].columns):
        if source in df_temporal_pivot['weekly_conversion_rate'].columns:
            # Filtrar valores cero para evitar líneas confusas
            data = df_temporal_pivot['weekly_conversion_rate'][source]
            data_filtered = data[data > 0]
            ax3.plot(data_filtered.index, data_filtered.values, 
                    marker=None, linewidth=2.5, markersize=6, label=source, color=colors[i])
    
    ax3.set_title('Evolución Semanal de Tasa de Conversión por Fuente', fontweight='bold')
    ax3.set_xlabel('Semana')
    ax3.set_ylabel('Tasa de Conversión (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Análisis de eficiencia (Inscripciones/Leads ratio)
    total_leads_weekly = df_temporal_pivot['weekly_leads'].sum(axis=1)
    total_inscr_weekly = df_temporal_pivot['weekly_inscriptions'].sum(axis=1)
    efficiency = (total_inscr_weekly / total_leads_weekly * 100).fillna(0)
    
    ax4.fill_between(df_temporal_pivot.index, efficiency, alpha=0.4, color='blue', label='Eficiencia Global')
    ax4.plot(df_temporal_pivot.index, efficiency, color='blue', linewidth=3, marker=None, markersize=8)
    
    # Añadir línea de tendencia
    x_numeric = np.arange(len(efficiency))
    valid_mask = ~np.isnan(efficiency) & (efficiency > 0)
    if valid_mask.sum() > 1:
        z = np.polyfit(x_numeric[valid_mask], efficiency[valid_mask], 1)
        trend = np.poly1d(z)
        ax4.plot(df_temporal_pivot.index, trend(x_numeric), '--', color='red', linewidth=2, 
                label=f'Tendencia: {"↗" if z[0] > 0 else "↘"} {z[0]:.3f}%/semana')
    
    # Añadir valores en puntos clave
    for i, (idx, val) in enumerate(efficiency.items()):
        if i % 3 == 0 and val > 0:  # Mostrar cada 3 semanas
            ax4.annotate(f'{val:.1f}%', (idx, val), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
    
    ax4.set_title('Eficiencia Global del Embudo (Inscripciones/Leads)', fontweight='bold')
    ax4.set_xlabel('Semana')
    ax4.set_ylabel('Eficiencia Global (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    temporal_file = os.path.join(output_path, f'temporal_analysis_{timestamp}.png')
    plt.savefig(temporal_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Guardado: {temporal_file}")

def create_campaign_performance_analysis(df_campaigns, output_path, timestamp):
    """
    Crear análisis de rendimiento de campañas individuales.

    Parámetros:
    df_campaigns : pd.DataFrame
        DataFrame con métricas de campaña:
        - total_leads
        - total_inscriptions
        - conversion_rate
        - campaign_cost
        - total_revenue
        - channel
        - course_type
    output_path : str
        Carpeta donde guardar los resultados.
    timestamp : str
        Marca de tiempo usada en el nombre del archivo.

    Retorna:
        - Gráfico PNG con:
        1. Bubble chart: Conversión vs Volumen (color = ROI, tamaño = volumen).
        2. Ranking de campañas por tasa de conversión.
    """
    print("Creando análisis de rendimiento por campaña...")
    
    # Filtrar campañas con datos suficientes
    df_filtered = df_campaigns[df_campaigns['total_leads'] >= 10].copy()
    
    if len(df_filtered) == 0:
        print("No hay suficientes datos de campañas para análisis detallado")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14))
    fig.suptitle('Análisis de Rendimiento por Campaña Individual', fontsize=16, fontweight='bold')
    
    # 1. Bubble chart: Conversión vs Volumen vs ROI
    df_filtered['roi'] = ((df_filtered['total_revenue'] - df_filtered['campaign_cost']) / 
                         df_filtered['campaign_cost'] * 100).fillna(0)
    
    # Normalizar tamaño de burbujas
    bubble_sizes = (df_filtered['total_leads'] / df_filtered['total_leads'].max() * 1000) + 100
    
    scatter = ax1.scatter(df_filtered['conversion_rate'], df_filtered['total_leads'], 
                         s=bubble_sizes, c=df_filtered['roi'], 
                         cmap='RdYlGn', alpha=0.9, edgecolors='black', linewidth=1)
    
    # Labels para todas las campañas
    for idx, row in df_filtered.iterrows():
        # Crear etiqueta más legible
        label = f"{row['channel']}-{row['course_type']}"
        ax1.annotate(label, (row['conversion_rate'], row['total_leads']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Líneas de referencia
    median_conversion = df_filtered['conversion_rate'].median()
    median_leads = df_filtered['total_leads'].median()
    ax1.axvline(x=median_conversion, color='red', linestyle='--', alpha=0.5, 
               label=f'Conversión Media: {median_conversion:.1f}%')
    ax1.axhline(y=median_leads, color='blue', linestyle='--', alpha=0.5,
               label=f'Leads Medianos: {median_leads:.0f}')
    
    ax1.set_xlabel('Tasa de Conversión (%)')
    ax1.set_ylabel('Volumen de Leads')
    ax1.set_title('Rendimiento por Campaña: Conversión vs Volumen\n(Tamaño = Volumen, Color = ROI)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Colorbar para ROI
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('ROI (%)')
    
    # 2. Ranking de campañas por eficiencia
    df_sorted = df_filtered.sort_values('conversion_rate', ascending=True)
    
    # Crear barras horizontales con colores por ROI
    bars = ax2.barh(range(len(df_sorted)), df_sorted['conversion_rate'], 
                   color=plt.cm.RdYlGn((df_sorted['roi'] - df_sorted['roi'].min()) / 
                                      (df_sorted['roi'].max() - df_sorted['roi'].min() + 0.001)))
    
    # Labels
    campaign_labels = [f"{row['channel']}-{row['course_type']}\n({row['total_leads']} leads)" 
                      for _, row in df_sorted.iterrows()]
    ax2.set_yticks(range(len(campaign_labels)), labels=campaign_labels)
    ax2.tick_params(axis='y', labelsize=10)
    
    # Valores en las barras
    for i, (bar, row) in enumerate(zip(bars, df_sorted.itertuples())):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{row.conversion_rate:.1f}%\nROI: {row.roi:.0f}%', 
                ha='left', va='center', fontweight='bold', fontsize=9)
    
    ax2.set_xlabel('Tasa de Conversión (%)')
    ax2.set_title('Ranking de Campañas por Tasa de Conversión', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    campaign_file = os.path.join(output_path, f'campaign_performance_{timestamp}.png')
    plt.savefig(campaign_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Guardado: {campaign_file}")

def generate_executive_summary(df_funnel, df_campaigns, df_temporal, output_path, timestamp):
    """Generar resumen ejecutivo del análisis de embudo
    
    Este módulo toma los resultados de los análisis previos y genera
    un reporte ejecutivo en formato de texto, que resume las métricas clave:
    
    - Totales globales del embudo (leads, inscripciones, conversión).
    - Identificación de la mejor y peor fuente de leads.
    - Campaña con mejor rendimiento (si hay datos).
    - Evolución temporal de la conversión (tendencia).
    - Recomendaciones estratégicas de optimización.
    
    Parámetros:
        df_funnel (pd.DataFrame): Resultados del embudo por fuente.
        df_campaigns (pd.DataFrame): Resultados de campañas.
        df_temporal (pd.DataFrame): Evolución temporal de leads/inscripciones.
        output_path (str): Carpeta donde guardar los resultados.
        timestamp (str): Marca de tiempo única del análisis.
    """
    print("Generando resumen ejecutivo del embudo...")
    
    # Métricas globales
    total_leads = df_funnel['total_leads'].sum()
    total_inscriptions = df_funnel['total_inscriptions'].sum()
    global_conversion = (total_inscriptions / total_leads * 100) if total_leads > 0 else 0
    
    # Mejor fuente
    best_source = df_funnel.loc[df_funnel['conversion_rate'].idxmax()]
    worst_source = df_funnel.loc[df_funnel['conversion_rate'].idxmin()]
    
    # Mejor campaña
    if len(df_campaigns) > 0:
        best_campaign = df_campaigns.loc[df_campaigns['conversion_rate'].idxmax()]
        campaign_info = f"""
MEJOR CAMPAÑA:
• Nombre: {best_campaign['campaign_name']}
• Canal: {best_campaign['channel']} - {best_campaign['course_type']}
• Conversión: {best_campaign['conversion_rate']:.2f}%
• Leads: {best_campaign['total_leads']:,}
• Inscripciones: {best_campaign['total_inscriptions']:,}"""
    else:
        campaign_info = "\nNo hay datos suficientes de campañas individuales."
    
    # Tendencia temporal
    if len(df_temporal) > 0:
        df_temp_summary = df_temporal.groupby('week_start').agg({
            'weekly_leads': 'sum',
            'weekly_inscriptions': 'sum'
        }).reset_index()
        df_temp_summary['conversion'] = (df_temp_summary['weekly_inscriptions'] / 
                                        df_temp_summary['weekly_leads'] * 100)
        
        first_month_conv = df_temp_summary.head(4)['conversion'].mean()
        last_month_conv = df_temp_summary.tail(4)['conversion'].mean()
        trend = "MEJORANDO" if last_month_conv > first_month_conv else "EMPEORANDO"
        trend_change = last_month_conv - first_month_conv
        
        temporal_info = f"""
TENDENCIA TEMPORAL:
• Tendencia de Conversión: {trend} ({trend_change:+.1f}%)
• Conversión Primer Mes: {first_month_conv:.1f}%
• Conversión Último Mes: {last_month_conv:.1f}%
• Total Semanas Analizadas: {len(df_temp_summary)}"""
    else:
        temporal_info = "\nNo hay suficientes datos temporales para análisis de tendencias."
    
    report = f"""
RESUMEN EJECUTIVO - ANÁLISIS DE EMBUDO DE CONVERSIÓN
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MÉTRICAS GLOBALES DEL EMBUDO:
• Total Leads: {total_leads:,}
• Total Inscripciones: {total_inscriptions:,}
• Conversión Global: {global_conversion:.2f}%

ANÁLISIS POR FUENTE:
• Mejor Fuente: {best_source['source_group']} ({best_source['conversion_rate']:.2f}%)
• Peor Fuente: {worst_source['source_group']} ({worst_source['conversion_rate']:.2f}%)
• Diferencia: {best_source['conversion_rate'] - worst_source['conversion_rate']:.2f} puntos porcentuales

{campaign_info}

{temporal_info}

RECOMENDACIONES ESTRATÉGICAS:
• Potenciar inversión en: {best_source['source_group']}
• Optimizar o reducir: {worst_source['source_group']}
• Monitorear evolución temporal para ajustar estrategias
• Enfocar esfuerzos en mejorar conversión de fuentes de alto volumen
"""
    
    # Guardar reporte
    report_file = os.path.join(output_path, f'funnel_executive_summary_{timestamp}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"Guardado: {report_file}")

def display_results_summary(df_funnel, df_campaigns):
    """Mostrar resumen de resultados en consola
    
    Se imprimen los principales resultados del análisis:
    - Embudo de conversión por fuente.
    - Top 10 campañas ordenadas por conversión (si hay datos disponibles).
    """
    print("\nRESULTADOS DEL EMBUDO POR FUENTE:")
    print(df_funnel.to_string(index=False))
    
    if len(df_campaigns) > 0:
        print("\nTOP 10 CAMPAÑAS POR CONVERSIÓN:")
        print(df_campaigns.head(10).to_string(index=False))
    else:
        print("\nNo hay datos suficientes de campañas individuales para mostrar.")

def main():
    """Función principal del análisis de embudo de conversión
    
    Flujo completo del script:
    1. Configurar rutas de trabajo.
    2. Conectar a la base de datos SQLite.
    3. Ejecutar queries y obtener resultados.
    4. Guardar resultados en CSV.
    5. Mostrar resumen de resultados en consola.
    6. Configurar estilo de visualizaciones.
    7. Crear visualizaciones del embudo de conversión.
    8. Crear análisis temporal.
    9. Crear análisis de rendimiento por campaña.
    10. Generar resumen ejecutivo.
    11. Cerrar conexión a la base de datos.
    """
    print("Iniciando análisis de embudo de conversión - Objetivo 2...")
    
    # 1. Configurar rutas
    db_path, output_path = setup_paths()
    
    # 2. Conectar a base de datos
    conn = connect_to_database(db_path)
    
    # 3. Ejecutar queries
    df_funnel, df_campaigns, df_temporal, df_quality = execute_funnel_queries(conn)
    
    # 4. Guardar resultados en CSV
    timestamp = save_results_to_csv(df_funnel, df_campaigns, df_temporal, df_quality, output_path)
    
    # 5. Mostrar resultados
    display_results_summary(df_funnel, df_campaigns)
    
    # 6. Configurar estilo de visualizaciones
    configure_plot_style()
    
    # 7. Crear visualizaciones del embudo
    create_funnel_visualization(df_funnel, output_path, timestamp)
    
    # 8. Crear análisis temporal
    create_temporal_analysis(df_temporal, output_path, timestamp)
    
    # 9. Crear análisis de rendimiento por campaña
    create_campaign_performance_analysis(df_campaigns, output_path, timestamp)
    
    # 10. Generar resumen ejecutivo
    generate_executive_summary(df_funnel, df_campaigns, df_temporal, output_path, timestamp)
    
    # 11. Cerrar conexión
    conn.close()
    
    print("\nANÁLISIS DE EMBUDO DE CONVERSIÓN COMPLETADO!")
    print(f"Todos los archivos guardados en: {output_path}")
    print("\nArchivos generados:")
    print("• funnel_analysis_[timestamp].csv - Datos del embudo por fuente")
    print("• campaign_conversion_[timestamp].csv - Conversión por campaña")
    print("• temporal_analysis_[timestamp].csv - Evolución temporal")
    print("• lead_quality_analysis_[timestamp].csv - Calidad de leads")
    print("• conversion_funnel_[timestamp].png - Visualización del embudo")
    print("• temporal_analysis_[timestamp].png - Análisis temporal")
    print("• campaign_performance_[timestamp].png - Rendimiento por campaña")
    print("• funnel_executive_summary_[timestamp].txt - Resumen ejecutivo")


if __name__ == "__main__":
    main()