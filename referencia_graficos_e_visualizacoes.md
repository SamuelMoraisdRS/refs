# 📊 Guia Prático & Modelo de Referência: Gráficos e Visualização de Dados
> **Ciência de Dados & Estatística Inferencial**  
> Este notebook é um **guia utilitário completo de código** para criação, customização e interpretação de gráficos em análises exploratórias (EDA), testes de hipóteses, diagnósticos de normalidade e avaliação de modelos de regressão.

---

## 1. Configuração de Ambiente e Estilização Padrão
Carregamento das bibliotecas principais (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats`) e preparação de estilos visuais limpos e profissionais.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configurações globais de estilo visual para relatórios e provas
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 120

# Carregamento de Datasets Didáticos das Aulas
# 1. mtcars (Automóveis)
mtcars_data = {
    'mpg': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4],
    'cyl': [6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 6, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 8, 6, 8, 4],
    'hp': [110, 110, 93, 110, 175, 105, 245, 62, 95, 123, 123, 180, 180, 180, 205, 215, 230, 66, 52, 65, 97, 150, 150, 245, 175, 66, 91, 113, 264, 175, 335, 109],
    'wt': [2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440, 3.440, 4.070, 3.730, 3.780, 5.250, 5.424, 5.345, 2.200, 1.615, 1.835, 2.465, 3.520, 3.435, 3.840, 3.845, 1.935, 2.140, 1.513, 3.170, 2.770, 3.570, 2.780],
    'am': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}
df_mtcars = pd.DataFrame(mtcars_data)
df_mtcars['transmissao'] = df_mtcars['am'].map({0: 'Automático', 1: 'Manual'})

# 2. sleep (Ensaio Clínico de Sono)
sleep_data = {
    'extra': [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0, 1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4],
    'group': [1]*10 + [2]*10,
    'ID': list(range(1, 11)) * 2
}
df_sleep = pd.DataFrame(sleep_data)

# 3. titanic (Passageiros e Sobrevivência)
np.random.seed(42)
df_titanic = pd.DataFrame({
    'pclass': np.random.choice([1, 2, 3], size=200, p=[0.25, 0.25, 0.50]),
    'sex': np.random.choice(['female', 'male'], size=200, p=[0.4, 0.6]),
    'fare': np.random.exponential(scale=30, size=200) + 10,
    'survived': np.random.choice([0, 1], size=200, p=[0.6, 0.4])
})

print("✅ Ambiente configurado e datasets de exemplo carregados com sucesso!")
```

## 2. Visualização de Distribuições Univariadas
Gráficos essenciais para inspeção da forma da distribuição, tendência central, dispersão, assimetria e detecção de *outliers*.

### 2.1 Histograma com Curva de Densidade (KDE) e Linhas de Média/Mediana
O **Histograma + KDE** permite avaliar a simetria dos dados. A comparação entre a média e a mediana ajuda a identificar se a distribuição é normal ou assimétrica.


```python
fig, ax = plt.subplots(figsize=(8, 5))

# Plot do histograma com KDE
sns.histplot(df_mtcars['mpg'], kde=True, bins=8, color='#0072b2', edgecolor='black', alpha=0.6, ax=ax)

# Cálculo de estatísticas descritivas
media_mpg = df_mtcars['mpg'].mean()
mediana_mpg = df_mtcars['mpg'].median()

# Adição de linhas verticais para Média e Mediana
ax.axvline(media_mpg, color='#d55e00', linestyle='--', linewidth=2, label=f'Média: {media_mpg:.2f} mpg')
ax.axvline(mediana_mpg, color='#009e73', linestyle='-', linewidth=2, label=f'Mediana: {mediana_mpg:.2f} mpg')

ax.set_title('Distribuição de Consumo de Combustível (mpg) - mtcars')
ax.set_xlabel('Milhas por Galão (mpg)')
ax.set_ylabel('Frequência / Contagem')
ax.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.show()
```

### 2.2 Boxplot Univariado e Agrupado por Categoria
O **Boxplot** exibe a mediana (linha central), o intervalo interquartil (IQR: Q1 a Q3), as hastes (*whiskers*) e os **outliers** (pontos além de 1.5 × IQR).


```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 1. Boxplot Univariado com orientação horizontal
sns.boxplot(x=df_mtcars['mpg'], color='#56b4e9', ax=ax1, width=0.4)
sns.stripplot(x=df_mtcars['mpg'], color='#222222', alpha=0.6, jitter=0.2, size=6, ax=ax1) # Pontos individuais
ax1.set_title('Boxplot do Consumo Geral (mpg)')
ax1.set_xlabel('Milhas por Galão (mpg)')

# 2. Boxplot Agrupado por Tipo de Transmissão
sns.boxplot(data=df_mtcars, x='transmissao', y='mpg', hue='transmissao', palette=['#e69f00', '#0072b2'], legend=False, ax=ax2, width=0.5)
sns.stripplot(data=df_mtcars, x='transmissao', y='mpg', color='black', alpha=0.5, jitter=0.2, ax=ax2)
ax2.set_title('Consumo (mpg) por Tipo de Transmissão')
ax2.set_xlabel('Tipo de Transmissão')
ax2.set_ylabel('Milhas por Galão (mpg)')

plt.tight_layout()
plt.show()
```

### 2.3 Violin Plot (Densidade + Quartis)
O **Violin Plot** combina a curva de densidade estimada nas laterais com a visualização dos quartis internamente, sendo ótimo para comparar distribuições bimodais.


```python
fig, ax = plt.subplots(figsize=(8, 5))

sns.violinplot(data=df_mtcars, x='cyl', y='hp', hue='cyl', palette='Set2', legend=False, inner='quartile', ax=ax)

ax.set_title('Distribuição de Potência (hp) por Número de Cilindros')
ax.set_xlabel('Número de Cilindros')
ax.set_ylabel('Potência (hp)')

plt.tight_layout()
plt.show()
```

### 2.4 Quantile-Quantile Plot (Q-Q Plot) para Validação de Normalidade
O **Q-Q Plot** é a ferramenta gráfica mais importante para diagnosticar a premissa de normalidade antes de aplicar testes paramétricos (como o Teste t ou ANOVA). Se os pontos seguirem a reta diagonal vermelha, a distribuição é aproximadamente normal.


```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 1. Q-Q Plot de variável com distribuição próxima da Normal
stats.probplot(df_mtcars['mpg'], dist="norm", plot=ax1)
ax1.get_lines()[0].set_markerfacecolor('#0072b2')
ax1.get_lines()[0].set_markeredgecolor('#0072b2')
ax1.get_lines()[1].set_color('#d55e00')
ax1.set_title('Q-Q Plot: mpg (mtcars) ~ Normal')
ax1.set_xlabel('Quantis Teóricos (Normal)')
ax1.set_ylabel('Quantis Observados')

# 2. Q-Q Plot de variável Assimétrica (Tarifa do Titanic)
stats.probplot(df_titanic['fare'], dist="norm", plot=ax2)
ax2.get_lines()[0].set_markerfacecolor('#e69f00')
ax2.get_lines()[0].set_markeredgecolor('#e69f00')
ax2.get_lines()[1].set_color('#d55e00')
ax2.set_title('Q-Q Plot: Tarifa (Titanic) ~ Assimétrica')
ax2.set_xlabel('Quantis Teóricos (Normal)')
ax2.set_ylabel('Quantis Observados')

plt.tight_layout()
plt.show()
```

## 3. Relações Numéricas, Regressão e Linha de Correlação de Pearson
Gráficos para avaliar a força, sentido e formato da associação entre duas variáveis numéricas contínuas.

### 3.1 Scatterplot com Reta de Regressão Linear e Intervalo de Confiança
Utilize `sns.regplot()` para traçar o gráfico de dispersão com a **reta de regressão OLS**, a banda de sombreamento do **Intervalo de Confiança de 95%** e a anotação do coeficiente de **Correlação de Pearson ($r$)** e do $p$-valor.


```python
fig, ax = plt.subplots(figsize=(8, 6))

# Plot de Dispersão com Regressão
sns.regplot(data=df_mtcars, x='wt', y='mpg',
            scatter_kws={'s': 60, 'color': '#0072b2', 'alpha': 0.8},
            line_kws={'color': '#d55e00', 'linewidth': 2, 'label': 'Reta OLS'},
            ax=ax)

# Cálculo da Correlação de Pearson para Anotação
r, p_val = stats.pearsonr(df_mtcars['wt'], df_mtcars['mpg'])

# Caixa de Anotação Estatística no Gráfico
texto_correlacao = f"Pearson r = {r:.4f}\np-valor = {p_val:.4e}\nR² = {r**2:.4f}"
ax.text(0.65, 0.80, texto_correlacao, transform=ax.transAxes,
        fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.9))

ax.set_title('Relação Linear: Peso do Veículo (wt) vs Consumo (mpg)')
ax.set_xlabel('Peso (1000 lbs)')
ax.set_ylabel('Milhas por Galão (mpg)')
ax.legend(loc='lower left')

plt.tight_layout()
plt.show()
```

```python
# Dispersão dos valores e dos postos, com a reta de mínimos quadrados
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

panels = [
    (mtcars["hp"], mtcars["mpg"], "Potência (hp)", "Consumo (mpg)", "Pearson"),
    (ranks_hp, ranks_mpg, "Posto da potência", "Posto do consumo", "Spearman"),
]

for ax, (x, y, xlabel, ylabel, name) in zip(axes, panels):
    slope, intercept = np.polyfit(x, y, deg=1)
    grid = np.linspace(x.min(), x.max(), 100)
    ax.scatter(x, y, color="#1F4E79", alpha=.8)
    ax.plot(grid, intercept + slope * grid, color="#DD8452", linewidth=2)
    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=f"{name}: {stats.pearsonr(x, y).statistic:.2f}")

plt.tight_layout()
plt.show()
```

```python

fig, ax = plt.subplots(figsize=(8, 5))

# Plot do histograma com KDE
sns.histplot(df_mtcars['mpg'], kde=True, bins=8, color='#0072b2', edgecolor='black', alpha=0.6, ax=ax)

# Cálculo de estatísticas descritivas
media_mpg = df_mtcars['mpg'].mean()
mediana_mpg = df_mtcars['mpg'].median()

# Adição de linhas verticais para Média e Mediana
ax.axvline(media_mpg, color='#d55e00', linestyle='--', linewidth=2, label=f'Média: {media_mpg:.2f} mpg')
ax.axvline(mediana_mpg, color='#009e73', linestyle='-', linewidth=2, label=f'Mediana: {mediana_mpg:.2f} mpg')

ax.set_title('Distribuição de Consumo de Combustível (mpg) - mtcars')
ax.set_xlabel('Milhas por Galão (mpg)')
ax.set_ylabel('Frequência / Contagem')
ax.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.show()
```
### 3.2 Scatterplot Agrupado por Variável Categórica (Hue & Style)
Permite observar a relação bivariada entre duas variáveis contínuas dividida por grupos ou categorias.


```python
fig, ax = plt.subplots(figsize=(9, 6))

sns.scatterplot(data=df_mtcars, x='hp', y='mpg', hue='transmissao', style='transmissao',
                s=90, palette=['#e69f00', '#0072b2'], ax=ax)

# Linhas de Regressão Separadas por Grupo (usando regplot individual)
for trans, col in zip(['Automático', 'Manual'], ['#e69f00', '#0072b2']):
    sub_df = df_mtcars[df_mtcars['transmissao'] == trans]
    sns.regplot(data=sub_df, x='hp', y='mpg', scatter=False, ax=ax, color=col,
                line_kws={'linestyle': '--', 'linewidth': 1.5, 'label': f'Reta {trans}'})

ax.set_title('Potência (hp) vs Consumo (mpg) por Tipo de Transmissão')
ax.set_xlabel('Potência (hp)')
ax.set_ylabel('Milhas por Galão (mpg)')
ax.legend(title='Transmissão')

plt.tight_layout()
plt.show()
```

### 3.3 Efeito da Transformação Logarítmica para Linearizar Relações Curvas
Quando a relação entre $X$ e $Y$ é não linear (curva), a **transformação $\log(X)$** ajuda a linearizar a relação, permitindo o uso da Correlação de Pearson e Regressão Linear Simples.


```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 1. Relação Não Linear Original (Curva)
r_orig, p_orig = stats.pearsonr(df_mtcars['hp'], df_mtcars['mpg'])
sns.regplot(data=df_mtcars, x='hp', y='mpg', color='#0072b2', ax=ax1)
ax1.set_title(f'Escala Original: hp vs mpg\nPearson r = {r_orig:.4f}')
ax1.set_xlabel('Potência Original (hp)')
ax1.set_ylabel('Consumo (mpg)')

# 2. Relação Linearizada com Log(hp)
log_hp = np.log(df_mtcars['hp'])
r_log, p_log = stats.pearsonr(log_hp, df_mtcars['mpg'])
sns.regplot(x=log_hp, y=df_mtcars['mpg'], color='#009e73', ax=ax2)
ax2.set_title(f'Escala Logarítmica: log(hp) vs mpg\nPearson r = {r_log:.4f}')
ax2.set_xlabel('Logarítmo da Potência: log(hp)')
ax2.set_ylabel('Consumo (mpg)')

plt.tight_layout()
plt.show()
```

## 4. Visualizações Categóricas e Tabelas de Contingência
Gráficos para contagens, proporções, comparação de médias categóricas e matrizes de frequência.

### 4.1 Barplot de Médias com Barras de Erro (Intervalo de Confiança de 95%)
O `sns.barplot()` calcula automaticamente a média de cada categoria e projeta a barra de erro referente ao **IC 95% da média**.


```python
fig, ax = plt.subplots(figsize=(8, 5))

# Barplot com IC 95% (errorbar='ci')
sns.barplot(data=df_mtcars, x='cyl', y='mpg', hue='transmissao', palette='Blues_d', errorbar=('ci', 95), ax=ax)

# Anotação de valores sobre as barras
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height) and height > 0:
        ax.annotate(f'{height:.1f}',
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=9, color='white', weight='bold')

ax.set_title('Consumo Média (mpg) por Cilindros e Transmissão (IC 95%)')
ax.set_xlabel('Número de Cilindros')
ax.set_ylabel('Média de Consumo (mpg)')
ax.legend(title='Transmissão')

plt.tight_layout()
plt.show()
```

### 4.2 Countplot e Gráfico de Proporções Categóricas
Exibe a contagem ou porcentagem de ocorrência em dados categóricos/binários (ideal para suporte a testes Qui-Quadrado ou Z).


```python
fig, ax = plt.subplots(figsize=(8, 5))

# Criando mapa de nomes para legibilidade do Titanic
df_titanic['Sobreviveu'] = df_titanic['survived'].map({0: 'Não', 1: 'Sim'})
df_titanic['Classe'] = df_titanic['pclass'].map({1: '1ª Classe', 2: '2ª Classe', 3: '3ª Classe'})

# Countplot agrupado
sns.countplot(data=df_titanic, x='Classe', hue='Sobreviveu', palette=['#d55e00', '#009e73'], ax=ax)

# Adicionando rótulos de contagem exata nas barras
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height + 1),
                    ha='center', va='bottom', fontsize=9, weight='bold')

ax.set_title('Contagem de Sobreviventes por Classe de Passagem (Titanic)')
ax.set_xlabel('Classe de Viagem')
ax.set_ylabel('Número de Passageiros')
ax.set_ylim(0, df_titanic['Classe'].value_counts().max() * 1.25)

plt.tight_layout()
plt.show()
```

### 4.3 Heatmap de Tabela de Contingência (Cruzamento de Categóricas)
Visualização gráfica das contagens observadas entre duas variáveis categóricas para apoiar a interpretação do **Teste Qui-Quadrado ($\chi^2$) de Independência**.


```python
fig, ax = plt.subplots(figsize=(7, 5))

# Tabela de contingência observada
tabela_cruzada = pd.crosstab(df_titanic['Classe'], df_titanic['Sobreviveu'])

# Plot do Heatmap
sns.heatmap(tabela_cruzada, annot=True, fmt='d', cmap='YlGnBu', cbar=True, linewidths=1, ax=ax)

ax.set_title('Tabela de Contingência: Classe vs Sobrevivência')
ax.set_xlabel('Sobreviveu?')
ax.set_ylabel('Classe do Passageiro')

plt.tight_layout()
plt.show()
```

## 5. Matrizes de Correlação e Exploração Multivariada
Ferramentas para visualização simultânea da correlação entre todas as variáveis numéricas do dataset.

### 5.1 Heatmap de Correlação (Pearson e Spearman)
O **Heatmap de Correlação** usa um mapa de cores divergente (`coolwarm` ou `RdBu_r`) para destacar associações positivas fortes (próximas de +1) e negativas fortes (próximas de -1).


```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Matriz de Correlação de Pearson
corr_pearson = df_mtcars[['mpg', 'hp', 'wt', 'cyl']].corr(method='pearson')
sns.heatmap(corr_pearson, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, ax=ax1)
ax1.set_title('Matriz de Correlação de Pearson (Linear)')

# 2. Matriz de Correlação de Spearman
corr_spearman = df_mtcars[['mpg', 'hp', 'wt', 'cyl']].corr(method='spearman')
sns.heatmap(corr_spearman, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, ax=ax2)
ax2.set_title('Matriz de Correlação de Spearman (Monotônica / Postos)')

plt.tight_layout()
plt.show()
```

### 5.2 Pairplot (Matriz de Dispersão e Distribuições Diagonais)
O `sns.pairplot()` gera uma grade completa de scatterplots para todos os pares de variáveis numéricas, colocando histogramas ou KDEs na diagonal principal.


```python
# Gerando Pairplot para variáveis chave do mtcars agrupadas por Transmissão
g = sns.pairplot(df_mtcars[['mpg', 'hp', 'wt', 'transmissao']], hue='transmissao', palette=['#e69f00', '#0072b2'], corner=False)
g.fig.suptitle('Pairplot Multivariado: Consumo, Potência e Peso', y=1.02, weight='bold')

plt.show()
```

## 6. Gráficos de Diagnóstico para Regressão Linear (Aula 10)
Visualizações fundamentais para diagnosticar as premissas dos resíduos em modelos de Regressão Linear OLS (Linearidade, Homocedasticidade e Normalidade dos Resíduos).


```python
# Ajuste de um Modelo de Regressão Linear Simples via NumPy/SciPy para obter os resíduos
x = df_mtcars['wt']
y = df_mtcars['mpg']
slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)

# Valores preditos (Fitted values) e Resíduos
y_pred = intercept + slope * x
residuos = y - y_pred

# Painel de Diagnóstico de Regressão (4 Gráficos)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Valores Ajustados vs Resíduos (Homocedasticidade e Linearidade)
sns.scatterplot(x=y_pred, y=residuos, color='#0072b2', s=70, ax=axes[0, 0])
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[0, 0].set_title('Resíduos vs Valores Ajustados (Fitted)')
axes[0, 0].set_xlabel('Valores Ajustados (Preditos)')
axes[0, 0].set_ylabel('Resíduos (Real - Predito)')

# 2. Q-Q Plot dos Resíduos (Normalidade dos Resíduos)
stats.probplot(residuos, dist="norm", plot=axes[0, 1])
axes[0, 1].get_lines()[0].set_markerfacecolor('#0072b2')
axes[0, 1].get_lines()[1].set_color('red')
axes[0, 1].set_title('Q-Q Plot dos Resíduos')

# 3. Histograma dos Resíduos
sns.histplot(residuos, kde=True, color='#009e73', ax=axes[1, 0])
axes[1, 0].axvline(0, color='red', linestyle='--')
axes[1, 0].set_title('Distribuição / Histograma dos Resíduos')
axes[1, 0].set_xlabel('Valor do Resíduo')

# 4. Resíduos vs Ordem das Observações (Independência)
axes[1, 1].plot(range(len(residuos)), residuos, marker='o', color='#e69f00', linestyle='-')
axes[1, 1].axhline(0, color='red', linestyle='--')
axes[1, 1].set_title('Resíduos por Ordem de Observação')
axes[1, 1].set_xlabel('Índice da Observação')
axes[1, 1].set_ylabel('Resíduo')

plt.tight_layout()
plt.show()
```
