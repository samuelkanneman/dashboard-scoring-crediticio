import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, roc_auc_score, auc
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")

# Configuración de la página
st.set_page_config(
    page_title="Tablero Riesgo Crédito",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    delimiter = ';'
    # Lista de encodings comunes
    encodings = ['utf-8','cp1252', 'latin1']

    for encoding in encodings:
        try:
            df = pd.read_csv('dataset_procesado.csv', delimiter=delimiter, encoding=encoding)
            print(f"Archivo leído correctamente con encoding: {encoding}")
            return df  # CORREGIDO: return dentro del try
        except UnicodeDecodeError:
            print(f"Error de decodificación con {encoding}")
            continue  # CORREGIDO: continue en lugar de return
    
    # Si ningún encoding funciona
    raise Exception("No se pudo leer el archivo con ningún encoding")

df = load_data()


# Función para calcular deciles
def calculate_deciles(df, score_column, recalculate=True):
    """
    Calcula deciles basados en el score
    Args:
        df: DataFrame con los datos
        score_column: Nombre de la columna de score a usar
        recalculate: Si True, recalcula deciles para la muestra actual
                    Si False, usa deciles precalculados del dataset original
    """
    df = df.copy()
    
    if recalculate:
        # Recalcular deciles para la muestra actual
        # Verificamos que tengamos suficientes datos únicos para 10 deciles
        unique_scores = df[score_column].nunique()
        
        if unique_scores >= 10:
            try:
                df['Decil'] = pd.qcut(df[score_column].rank(method='first'), 10, labels=False) + 1
                df['Decil'] = df['Decil'].astype(int)
            except ValueError:
                # Si qcut falla, usar percentiles manuales
                percentiles = np.percentile(df[score_column], np.linspace(0, 100, 11))
                df['Decil'] = pd.cut(df[score_column], bins=percentiles, labels=False, include_lowest=True) + 1
                df['Decil'] = df['Decil'].fillna(1).astype(int)
        else:
            # Si no hay suficientes valores únicos, crear deciles simples
            df['Decil'] = pd.cut(df[score_column], bins=10, labels=False) + 1
            df['Decil'] = df['Decil'].fillna(1).astype(int)
    else:
        # Usar deciles globales - si no existe la columna, crearla
        decil_global_col = f'Decil_Global_{score_column}'
        if decil_global_col not in df.columns:
            st.warning(f"⚠️ No se encontraron deciles globales para {score_column}. Calculando deciles para la muestra actual.")
            df['Decil'] = pd.qcut(df[score_column].rank(method='first'), 10, labels=False) + 1
            df['Decil'] = df['Decil'].astype(int)
        else:
            df['Decil'] = df[decil_global_col]
    
    return df


# Función para calcular métricas KS y AUC - MEJORADA
def calculate_metrics(df, score_column, recalc_deciles=True):
    """Calcula KS y AUC por período con opción de recalcular deciles"""
    metrics = []
    
    for periodo in df['periodo'].unique():
        df_periodo = df[df['periodo'] == periodo].copy()
        
        # Aplicar cálculo de deciles
        df_periodo = calculate_deciles(df_periodo, score_column, recalculate=recalc_deciles)
        
        # KS
        df_periodo['good'] = 1 - df_periodo['Malo']
        
        # Agrupar por decil para cálculo KS
        agg = df_periodo.groupby('Decil').agg(
            bads=('Malo', 'sum'),
            goods=('good', 'sum')
        ).reset_index()
        
        # Calcular acumulados
        total_bads = df_periodo['Malo'].sum()
        total_goods = df_periodo['good'].sum()
        
        if total_bads > 0 and total_goods > 0:
            agg['cum_bads'] = agg['bads'].cumsum() / total_bads
            agg['cum_goods'] = agg['goods'].cumsum() / total_goods
            agg['ks'] = np.abs(agg['cum_bads'] - agg['cum_goods']) * 100
            ks_max = agg['ks'].max()
        else:
            ks_max = 0
        
        # AUC
        try:
            df_periodo['Probabilidad'] = 1 / (1 + np.exp(df_periodo[score_column] / 1000))
            auc_score = roc_auc_score(df_periodo['Malo'], df_periodo['Probabilidad'])
        except:
            auc_score = 0.5  # AUC neutral si hay error
        
        metrics.append({
            'periodo': periodo,
            'KS': round(ks_max, 2),
            'AUC': round(auc_score * 100, 2),
            'Total_Registros': len(df_periodo),
            'Modelo': score_column,
            'Método': 'Recalculado' if recalc_deciles else 'Global'
        })
    
    return pd.DataFrame(metrics)


# SIDEBAR
st.sidebar.header("🔧 Configuración")

# Filtros
selected_periods = st.sidebar.multiselect(
    "📅 Seleccionar Períodos:",
    options=df['periodo'].unique(),
    default=df['periodo'].unique()
)

st.sidebar.markdown("---")

# SELECCIÓN DE MODELO - Detectar automáticamente las columnas de score
score_columns = ['ML_Score', 'VZ_score_3t','NS_scoret','SI_Scoret', 'Score_Alt','Nos_Fintech']

# Si no encuentra columnas automáticamente, mostrar todas las numéricas
if not score_columns:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    score_columns = [col for col in numeric_columns if col not in ['Malo', 'periodo', 'Decil', 'ClienteNuevo']]

if len(score_columns) > 1:
    selected_score_column = st.sidebar.selectbox(
        "🎯 Seleccionar Modelo de Score:",
        options=score_columns,
        index=0,
        help="Selecciona qué modelo de scoring analizar"
    )
else:
    selected_score_column = score_columns[0] if score_columns else 'sca_cad_res'
    st.sidebar.info(f"📊 Modelo actual: {selected_score_column}")

# Mostrar información del modelo seleccionado
modelo_info = df[selected_score_column].describe()
st.sidebar.markdown("---")


cliente_tipo = st.sidebar.radio(
    "👥 Tipo de Cliente:",
    options=["Todos", "Nuevos", "Existente"],
    index=0
)

# NUEVA OPCIÓN: Método de cálculo de deciles
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Configuración de Análisis")

recalc_deciles = st.sidebar.radio(
    "🎯 Método de Cálculo de Deciles:",
    options=["Recalcular por muestra", "Mantener deciles globales"],
    index=0,
    help="""
    **Recalcular por muestra**: Cada decil contiene exactamente 10% de la muestra filtrada.
    Ideal para analizar la performance del modelo en el segmento específico.
    
    **Mantener deciles globales**: Usa los rangos de score del dataset completo.
    Ideal para comparar cómo se distribuyen los segmentos en la escala original del modelo.
    """
)

# Mostrar información sobre el método seleccionado
if recalc_deciles == "Recalcular por muestra":
    st.sidebar.info("ℹ️ Los deciles se recalcularán para cada filtro aplicado. Cada decil tendrá ~10% de la muestra.")
else:
    st.sidebar.info("ℹ️ Se mantienen los deciles originales del modelo. Los deciles pueden tener diferente cantidad de registros.")

# Variable para el método
usar_deciles_recalculados = (recalc_deciles == "Recalcular por muestra")

# Primero, calcular deciles globales para el modelo seleccionado
decil_global_col = f'Decil_Global_{selected_score_column}'
if decil_global_col not in df.columns:
    # Calcular deciles globales una sola vez para todo el dataset con el modelo seleccionado
    df[decil_global_col] = pd.qcut(df[selected_score_column].rank(method='first'), 10, labels=False) + 1
    df[decil_global_col] = df[decil_global_col].astype(int)

# Filtrar datos
df_filtered = df[df['periodo'].isin(selected_periods)].copy()

# CORRECCIÓN: Usar los valores correctos para filtrar
if cliente_tipo == "Nuevos":
    df_filtered = df_filtered[df_filtered['ClienteNuevo'] == 1]
elif cliente_tipo == "Existente":  # CORREGIDO: cambiado de "Recurrentes" a "Existente"
    df_filtered = df_filtered[df_filtered['ClienteNuevo'] == 0]

# Aplicar el método de deciles seleccionado
if usar_deciles_recalculados:
    # Recalcular deciles para la muestra filtrada
    df_filtered = calculate_deciles(df_filtered, selected_score_column, recalculate=True)
    st.sidebar.success(f"✅ Deciles recalculados para {len(df_filtered)} registros")
else:
    # Usar deciles globales
    df_filtered['Decil'] = df_filtered[decil_global_col]
    st.sidebar.success(f"✅ Usando deciles globales para {len(df_filtered)} registros")

# DEBUG: Mostrar información sobre los deciles
if st.sidebar.checkbox("🔍 Mostrar información de debug"):
    st.sidebar.markdown("**Debug - Información de Deciles:**")
    
    if usar_deciles_recalculados:
        # Mostrar distribución de deciles recalculados
        decil_counts = df_filtered['Decil'].value_counts().sort_index()
        st.sidebar.text("Distribución deciles recalculados:")
        for decil, count in decil_counts.items():
            pct = count/len(df_filtered)*100
            st.sidebar.text(f"Decil {decil}: {count} ({pct:.1f}%)")
    else:
        # Mostrar cómo se distribuyen en deciles globales
        decil_counts = df_filtered['Decil'].value_counts().sort_index()
        st.sidebar.text("Distribución en deciles globales:")
        for decil, count in decil_counts.items():
            pct = count/len(df_filtered)*100
            st.sidebar.text(f"Decil {decil}: {count} ({pct:.1f}%)")

# HEADER
st.title("📊 Tablero de Scoring Crediticio CA")
st.markdown("---")

# MÉTRICAS PRINCIPALES
col1, col2, col3, col4 = st.columns(4)

total_clientes = len(df_filtered)
mora_pct = round(len(df_filtered[df_filtered['Malo'] == 1]) / total_clientes * 100, 2) if total_clientes > 0 else 0

# CORRECCIÓN: Calcular proporción de clientes nuevos vs Existente
if total_clientes > 0:
    nuevos_count = len(df_filtered[df_filtered['ClienteNuevo'] == 1])
    existente_count = len(df_filtered[df_filtered['ClienteNuevo'] == 0])  # CORREGIDO: nombre de variable
    nuevos_pct = round(nuevos_count / total_clientes * 100, 1)
    existente_pct = round(existente_count / total_clientes * 100, 1)  # CORREGIDO: usar existente_count
    
    # Crear el texto de la métrica dependiendo del filtro seleccionado
    if cliente_tipo == "Todos":
        client_ratio_label = "👥 Nuevos / Existente"
        client_ratio_value = f"{nuevos_pct}% / {existente_pct}%"
        client_ratio_delta = f"{nuevos_count:,} / {existente_count:,}"
    elif cliente_tipo == "Nuevos":
        client_ratio_label = "🆕 Clientes Nuevos"
        client_ratio_value = f"{nuevos_pct}%"
        client_ratio_delta = f"{nuevos_count:,} clientes"
    else:  # Existente
        client_ratio_label = "🔄 Clientes Existente"
        client_ratio_value = f"{existente_pct}%"
        client_ratio_delta = f"{existente_count:,} clientes"
else:
    client_ratio_label = "👥 Nuevos / Existente"
    client_ratio_value = "0% / 0%"
    client_ratio_delta = "0 / 0"

periodos_analizados = len(selected_periods)

with col1:
    st.metric(
        label="👥 Total Clientes",
        value=f"{total_clientes:,}",
        delta=None
    )

with col2:
    st.metric(
        label="⚠️ Morosidad (+60 días)",
        value=f"{mora_pct}%",
        delta=None
    )

with col3:
    st.metric(
        label=client_ratio_label,
        value=client_ratio_value,
        delta=client_ratio_delta
    )

with col4:
    st.metric(
        label="📅 Períodos Analizados",
        value=periodos_analizados,
        delta=None
    )

st.markdown("---")

# GRÁFICOS PRINCIPALES
tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribuciones", "🎯 Análisis por Deciles", "📈 Curvas de Performance", "📋 Métricas Detalladas"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de Score
        fig_dist = px.histogram(
            df_filtered, 
            x=selected_score_column,
            nbins=50,
            color='ClienteNuevo',
            title=f'Distribución del Score - {selected_score_column}',
            labels={selected_score_column: 'Score', 'count': 'Frecuencia'}
        )
        fig_dist.update_layout(showlegend=True)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Análisis de Nivel de Riesgo vs Morosidad
        if 'NivelRiesgo' in df_filtered.columns:
            # Calcular proporción de malos por nivel de riesgo
            risk_analysis = df_filtered.groupby(['NivelRiesgo', 'Malo']).size().unstack(fill_value=0)
            risk_analysis['Total'] = risk_analysis.sum(axis=1)
            risk_analysis['Porcentaje_Malos'] = (risk_analysis.get(1, 0) / risk_analysis['Total']) * 100
            risk_analysis['Count_Buenos'] = risk_analysis.get(0, 0)
            risk_analysis['Count_Malos'] = risk_analysis.get(1, 0)
            
            # Crear gráfico de barras
            fig_risk = go.Figure()
            
            # Barras para buenos
            fig_risk.add_trace(go.Bar(
                name='Buenos',
                x=risk_analysis.index,
                y=risk_analysis['Count_Buenos'],
                marker_color='lightblue'
            ))
            
            # Barras para malos
            fig_risk.add_trace(go.Bar(
                name='Malos',
                x=risk_analysis.index,
                y=risk_analysis['Count_Malos'],
                marker_color='salmon'
            ))
            
            # Añadir porcentajes como texto encima de las barras
            for i, (nivel, pct) in enumerate(zip(risk_analysis.index, risk_analysis['Porcentaje_Malos'])):
                fig_risk.add_annotation(
                    x=nivel,
                    y=risk_analysis.iloc[i]['Total'],
                    text=f"<b>{pct:.1f}%</b>",
                    showarrow=False,
                    yshift=10,
                    font=dict(size=12, color='yellow')  
                )
            
            fig_risk.update_layout(
                title=f'Distribución y % de Morosidad por Nivel de Riesgo',
                xaxis_title='Nivel de Riesgo',
                yaxis_title='Cantidad de Clientes',
                barmode='stack',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_risk, use_container_width=True)
            
        else:
            # Fallback: Si no existe NivelRiesgo, mostrar el gráfico original pero mejorado
            st.warning("⚠️ No se encontró la columna 'NivelRiesgo'. Mostrando distribución por período.")
            
            fig_periodo = px.histogram(
                df_filtered,
                x=selected_score_column,
                color='periodo',
                nbins=20,
                title=f'Distribución del Score por Período - {selected_score_column}',
                labels={selected_score_column: 'Score', 'count': 'Frecuencia'},
                opacity=0.7
            )
            st.plotly_chart(fig_periodo, use_container_width=True)

with tab2:
    # Análisis por deciles - MEJORADO
    col1, col2 = st.columns(2)
    
    with col1:
        # Porcentaje de malos por decil
        decil_analysis = df_filtered.groupby(['Decil', 'Malo']).size().unstack(fill_value=0)
        decil_analysis['Total'] = decil_analysis.sum(axis=1)
        decil_analysis['Porcentaje_Malos'] = (decil_analysis[1] / decil_analysis['Total']) * 100
        
        fig_decil = go.Figure()
        
        # Verificar si existen las columnas
        if 0 in decil_analysis.columns:
            fig_decil.add_trace(go.Bar(
                name='Buenos',
                x=decil_analysis.index,
                y=decil_analysis[0],
                marker_color='skyblue'
            ))
        
        if 1 in decil_analysis.columns:
            fig_decil.add_trace(go.Bar(
                name='Malos',
                x=decil_analysis.index,
                y=decil_analysis[1],
                marker_color='orange'
            ))
        
        # Añadir porcentajes como texto
        for i, pct in enumerate(decil_analysis['Porcentaje_Malos']):
            fig_decil.add_annotation(
                x=decil_analysis.index[i],
                y=decil_analysis.iloc[i]['Total'],
                text=f"{pct:.1f}%",
                showarrow=False,
                yshift=10
            )
        
        title_suffix = f"({recalc_deciles})"
        fig_decil.update_layout(
            title=f'Distribución de Buenos vs Malos por Decil {title_suffix}',
            xaxis_title='Decil',
            yaxis_title='Cantidad',
            barmode='stack'
        )
        
        st.plotly_chart(fig_decil, use_container_width=True)
    
    with col2:
        # Tendencia por período - MEJORADO
        periodo_decil_data = []
        
        for periodo in selected_periods:
            df_periodo = df_filtered[df_filtered['periodo'] == periodo].copy()
            if len(df_periodo) == 0:
                continue
            
            # Aplicar mismo método de deciles
            df_periodo = calculate_deciles(df_periodo, selected_score_column, recalculate=usar_deciles_recalculados)
            
            decil_periodo_temp = (
                df_periodo
                .groupby(['Decil', 'Malo'])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )
            
            if not decil_periodo_temp.empty:
                decil_periodo_temp['Total'] = decil_periodo_temp.get(0, 0) + decil_periodo_temp.get(1, 0)
                decil_periodo_temp['Porcentaje_Malos'] = (decil_periodo_temp.get(1, 0) / decil_periodo_temp['Total']) * 100
                decil_periodo_temp['periodo'] = periodo
                periodo_decil_data.append(decil_periodo_temp[['Decil', 'Porcentaje_Malos', 'periodo']])
        
        if periodo_decil_data:
            decil_periodo = pd.concat(periodo_decil_data, ignore_index=True)
            
            fig_trend = px.line(
                decil_periodo,
                x='Decil',
                y='Porcentaje_Malos',
                color='periodo',
                title=f'Evolución del % de Malos por Decil y Período ({recalc_deciles})',
                markers=True
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para mostrar la tendencia por período.")

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # Curva KS - MEJORADA
        fig_ks = go.Figure()
        
        for periodo in selected_periods:
            df_periodo = df_filtered[df_filtered['periodo'] == periodo].copy()
            if len(df_periodo) == 0:
                continue
            
            # Aplicar método de deciles consistente
            df_periodo = calculate_deciles(df_periodo, selected_score_column, recalculate=usar_deciles_recalculados)
            df_periodo['good'] = 1 - df_periodo['Malo']
            
            agg = df_periodo.groupby('Decil').agg(
                bads=('Malo', 'sum'),
                goods=('good', 'sum')
            ).reset_index()
            
            total_bads = df_periodo['Malo'].sum()
            total_goods = df_periodo['good'].sum()
            
            if total_bads > 0 and total_goods > 0:
                agg['cum_bads'] = agg['bads'].cumsum() / total_bads
                agg['cum_goods'] = agg['goods'].cumsum() / total_goods
                
                fig_ks.add_trace(go.Scatter(
                    x=agg['Decil'],
                    y=agg['cum_bads'],
                    mode='lines+markers',
                    name=f'Cum Bads {periodo}'
                ))
                
                fig_ks.add_trace(go.Scatter(
                    x=agg['Decil'],
                    y=agg['cum_goods'],
                    mode='lines+markers',
                    name=f'Cum Goods {periodo}'
                ))
        
        fig_ks.update_layout(
            title=f'Curvas KS por Período - {selected_score_column} ({recalc_deciles})',
            xaxis_title='Decil',
            yaxis_title='Proporción Acumulada'
        )
        
        st.plotly_chart(fig_ks, use_container_width=True)
    
    with col2:
        # Curva ROC - sin cambios necesarios
        fig_roc = go.Figure()
        
        for periodo in selected_periods:
            df_periodo = df_filtered[df_filtered['periodo'] == periodo].copy()
            if len(df_periodo) == 0:
                continue
                
            y_true = df_periodo['Malo']
            y_score = 1 / (1 + np.exp(df_periodo[selected_score_column] / 1000))
            
            try:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                
                fig_roc.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'{periodo} (AUC = {roc_auc:.3f})',
                    line=dict(width=2)
                ))
            except Exception as e:
                st.warning(f"Error calculando ROC para período {periodo}: {str(e)}")
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Aleatorio',
            line=dict(dash='dash', color='gray')
        ))
        
        fig_roc.update_layout(
            title=f'Curvas ROC por Período - {selected_score_column}',
            xaxis_title='Tasa de Falsos Positivos (FPR)',
            yaxis_title='Tasa de Verdaderos Positivos (TPR)'
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        # Tabla de métricas KS/AUC - MEJORADA
        metrics_table = calculate_metrics(df_filtered, selected_score_column, recalc_deciles=usar_deciles_recalculados)
        
        st.subheader("📊 Métricas de Performance")
        st.dataframe(
            metrics_table,
            use_container_width=True,
            hide_index=True
        )
        
        # Explicación del método
        if usar_deciles_recalculados:
            st.info("ℹ️ **Deciles Recalculados**: Cada decil contiene ~10% de la muestra filtrada. Las métricas reflejan la capacidad discriminatoria específica en este segmento.")
        else:
            st.info("ℹ️ **Deciles Globales**: Se mantienen los rangos originales del modelo. Las métricas muestran cómo se comporta el segmento en la escala completa del score.")
    
    with col2:
        # Información de deciles - MEJORADA
        st.subheader("📈 Información por Deciles")
        
        decil_info = df_filtered.groupby('Decil').agg({
            selected_score_column: ['min', 'max', 'count'],
            'Malo': 'sum'
        }).round(2)
        
        decil_info.columns = ['Min_Score', 'Max_Score', 'Total_Registros', 'Total_Malos']
        decil_info['Porcentaje_Malos'] = (decil_info['Total_Malos'] / decil_info['Total_Registros'] * 100).round(2)
        
        st.dataframe(
            decil_info,
            use_container_width=True
        )
        
        # Estadísticas adicionales
        st.markdown(f"**Estadísticas de la Muestra ({selected_score_column}):**")
        total_deciles = len(decil_info)
        promedio_registros = decil_info['Total_Registros'].mean()
        
        st.text(f"Deciles disponibles: {total_deciles}")
        st.text(f"Registros promedio por decil: {promedio_registros:.0f}")

# FOOTER
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        📊 Demo Tablero de Scoring Crediticio | Última actualización: {fecha}<br>
        Kanneman, Samuel | Especialista de Créditos
    </div>
    """.format(fecha=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")),
    unsafe_allow_html=True
)
