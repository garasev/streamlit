import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff  

df = pd.read_csv('./cleaned.csv')

sns.set_context("talk")  

st.title('Разведочный анализ данных банковских клиентов')

st.subheader('Первые строки данных')
st.write(df.head())

option = st.selectbox('Какую колонку вы хотите исследовать?', df.columns)

st.subheader('Распределение выбранной колонки')
sns.histplot(df[option], kde=True)
st.pyplot(plt)
plt.clf()  

st.subheader('Интерактивная матрица корреляций')
selected_corr_columns = st.multiselect('Выберите колонки для матрицы корреляций', df.columns.tolist(), default=df.columns.tolist())
corr = df[selected_corr_columns].corr()

fig = ff.create_annotated_heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale='Viridis',
    annotation_text=corr.round(2).values,
    showscale=True
)
st.plotly_chart(fig, use_container_width=True)

st.subheader('Распределения числовых признаков')
numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
selected_numerical = st.multiselect('Выберите числовые колонки для визуализации', numerical_columns, default=numerical_columns[0])

for col in selected_numerical:
    st.subheader(f'Распределение для {col}')
    sns.histplot(df[col], kde=True)
    st.pyplot(plt)
    plt.clf()  

st.subheader('Зависимости целевой переменной от признаков')
feature = st.selectbox('Выберите признак для сравнения с TARGET', df.columns.drop('TARGET'), index=0)

if df[feature].dtype == 'object' or len(df[feature].unique()) < 10:
    sns.countplot(x=feature, hue='TARGET', data=df)
else:
    sns.boxplot(x='TARGET', y=feature, data=df)
st.pyplot(plt)
plt.clf()  

st.subheader('Основные статистические характеристики числовых признаков')
selected_statistics = st.multiselect('Выберите числовые колонки для статистики', numerical_columns, default=numerical_columns[0])
st.write(df[selected_statistics].describe().transpose())

st.subheader('Визуализация зависимости между двумя признаками')
col1, col2 = st.selectbox('Выберите первый признак', df.columns, index=1), st.selectbox('Выберите второй признак', df.columns, index=2)

if len(df[col1].unique()) < 10 or len(df[col2].unique()) < 10:
    st.write('Для категориальных признаков или признаков с малым количеством уникальных значений рекомендуется использовать другие виды графиков.')
else:
    sns.scatterplot(x=col1, y=col2, data=df, hue='TARGET', alpha=0.6)
    st.pyplot(plt)
    plt.clf() 