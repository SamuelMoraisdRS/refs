# 🐍 Guia Prático & Modelo de Referência: Testes Estatísticos e Regressão Linear
Este notebook foi estruturado como um **guia completo de consulta rápida para provas práticas de código** e análises estatísticas em Ciência de Dados em Python.

---

### 📋 Estrutura da Referência:
1. **Configuração & Funções Utilitárias**: Imports essenciais, temas visuais e funções de suporte para formatação de testes.
2. **Estatística Inferencial & Teste de Permutação (Aula 7)**:
   - Efeito observado ($\Delta_{obs}$), Teste de Permutação empírico vs. Teste $t$.
   - Tamanho de Efeito (Cohen's $d$) e Correção de Bonferroni.
3. **Testes Paramétricos & Diagnóstico de Premissas (Aula 8)**:
   - Verificação de Normalidade (Shapiro-Wilk, Q-Q Plot) e Homocedasticidade (Levene).
   - Teste $t$ de 1 Amostra, Teste $t$ de Welch (2 amostras indep.), Teste $t$ Pareado.
   - ANOVA One-Way + Post-Hoc Tukey HSD.
   - Correlação de Pearson.
   - Teste $Z$ para 2 Proporções e Teste Qui-Quadrado ($\chi^2$) de Independência.
4. **Testes Não Paramétricos baseados em Postos (Aula 9)**:
   - Transformação e atribuição de Postos (Ranks).
   - Teste de Mann-Whitney U (Aproximação Normal & $\delta$ de Cliff).
   - Teste de Wilcoxon (Amostras Pareadas).
   - Teste de Kruskal-Wallis (Substituto da ANOVA).
   - Correlação de Spearman (Relações Monotônicas).
5. **Regressão Linear & Aprendizado de Máquina (Aula 10)**:
   - Protocolo de Validação: Divisão Treino / Teste Cego (`train_test_split`).
   - Regressão Linear Simples e Múltipla com `scikit-learn`.
   - Métricas de Desempenho ($MAE$, $RMSE$, $R^2$).
   - Validação Cruzada $K$-Fold ($5$ dobras) e estabilidade de coeficientes.
   - Diagnóstico por Gráficos de Resíduos.
   - Regularização: Ridge ($L_2$), Lasso ($L_1$) e ElasticNet com `Pipeline` e `StandardScaler`.


## 1. Configuração do Ambiente & Funções Utilitárias



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# Bibliotecas de Aprendizado de Máquina (Scikit-Learn)
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Configuração visual padrão
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams['figure.figsize'] = (8, 4.5)
plt.rcParams['font.size'] = 10

print("✅ Bibliotecas carregadas com sucesso!")

```


```python
def report_test(name, result, stat_label="Estatística", df=None):
    # Função utilitária para imprimir o resultado de testes estatísticos de forma padronizada.
    if isinstance(result, tuple):
        value, p = result[0], result[1]
    else:
        value, p = result.statistic, result.pvalue
    
    df_val = df if df is not None else getattr(result, "df", None)
    
    parts = [f"{stat_label} = {value:.4f}"]
    if df_val is not None:
        parts.append(f"gl = {df_val:.4g}" if np.isscalar(df_val) else f"gl = {df_val}")
    
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.4f}"
    parts.append(p_str)
    
    if hasattr(result, "confidence_interval"):
        ci = result.confidence_interval()
        parts.append(f"IC 95% [{ci.low:.4f} ; {ci.high:.4f}]")
        
    print(f" [{name}] " + " · ".join(parts))

```

## 2. Estatística Inferencial & Teste de Permutação (Aula 7)

### 2.1 Teste de Permutação Empírico
Mede o efeito observado $\Delta_{obs} = \bar{X}_A - \bar{X}_B$ e constrói a distribuição nula trocando aleatoriamente os rótulos dos grupos.



```python
# Exemplo de Simulação de Dados
rng = np.random.default_rng(42)

# Dados de Exemplo: Consumo (mpg) de carros com câmbio Manual vs Automático
group_manual = np.array([21.0, 21.0, 22.8, 24.4, 22.8, 30.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2])
group_auto = np.array([21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5])

delta_obs = group_manual.mean() - group_auto.mean()

def permutation_test_diff(g1, g2, n_perm=10000, seed=42):
    prng = np.random.default_rng(seed)
    combined = np.concatenate([g1, g2])
    n1 = len(g1)
    
    null_diffs = []
    for _ in range(n_perm):
        shuffled = prng.permutation(combined)
        d = shuffled[:n1].mean() - shuffled[n1:].mean()
        null_diffs.append(d)
        
    null_diffs = np.array(null_diffs)
    p_val = np.mean(np.abs(null_diffs) >= np.abs(delta_obs))
    return null_diffs, p_val

null_dist, p_val_perm = permutation_test_diff(group_manual, group_auto)

print(f"Δ_obs (Manual - Auto) = {delta_obs:.2f} mpg")
print(f"p-valor (Permutação Bilateral): {p_val_perm:.4f}")

# Visualização da Distribuição Nula
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(null_dist, bins=50, color="#A1C9F4", edgecolor="white", density=True, label="Distribuição Nula (Permutação)")
ax.axvline(delta_obs, color="#DD8452", linewidth=2, linestyle="--", label=f"+Δ_obs ({delta_obs:.2f})")
ax.axvline(-delta_obs, color="#DD8452", linewidth=2, linestyle="--", label=f"-Δ_obs ({-delta_obs:.2f})")
ax.set(title="Teste de Permutação - Distribuição sob H0", xlabel="Diferença de Médias (mpg)", ylabel="Densidade")
ax.legend()
plt.tight_layout()
plt.close()

```

### 2.2 Tamanho do Efeito (Cohen's $d$) & Correção de Bonferroni para Múltiplos Testes



```python
def cohens_d(g1, g2):
    # Calcula o d de Cohen (diferença de médias padronizada).
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(g1) - np.mean(g2)) / s_pooled

d_val = cohens_d(group_manual, group_auto)
print(f"Tamanho do Efeito (Cohen's d): {d_val:.4f} (Valores > 0.8 indicam efeito grande)")

# Correção de Bonferroni para k testes simultâneos
alpha_original = 0.05
k_testes = 10
alpha_bonferroni = alpha_original / k_testes
print(f"Alpha ajustado por Bonferroni (para {k_testes} testes): {alpha_bonferroni:.4f}")

```

## 3. Testes Paramétricos & Diagnóstico de Premissas (Aula 8)

### 3.1 Verificação de Premissas (Normalidade & Homocedasticidade)
- **Shapiro-Wilk**: Testa se a amostra vem de uma população Normal ($H_0$: Dados Normais).
  - Se rejeitar $H_0$: normalize com log ou faça um teste não-paramétrico
- **Levene**: Testa igualdade de variâncias entre grupos ($H_0$: Variâncias Iguais).
- **Q-Q Plot**: Diagnóstico visual de normalidade.



```python
# Verificação de Normalidade
shapiro_m = stats.shapiro(group_manual)
shapiro_a = stats.shapiro(group_auto)
report_test("Shapiro-Wilk (Manual)", shapiro_m, stat_label="W")
report_test("Shapiro-Wilk (Auto)", shapiro_a, stat_label="W")

# Verificação de Homocedasticidade
levene_test = stats.levene(group_manual, group_auto)
report_test("Levene (Igualdade de Variâncias)", levene_test, stat_label="Estatística W")

# Visualização Q-Q Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
stats.probplot(group_manual, dist="norm", plot=axes[0])
axes[0].set_title("Q-Q Plot: Grupo Manual")

stats.probplot(group_auto, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot: Grupo Automático")
plt.tight_layout()
plt.show()

```

### 3.2 Testes $t$ de Student e Welch

#### 1. Teste $t$ de 1 Amostra
Compara a média de uma amostra com um valor de referência $\mu_0$.
**Premissas**
- Amostras independentes
- Distribuição aproximadamente normal ou $n < 30$
**Hipóteses**
- $H_0$: A média de x é igual ao alvo
- $H_1$: A média de x é diferente do alvo



```python
# Teste t de 1 Amostra
target_val = 20.0
t_1samp = stats.ttest_1samp(group_manual, popmean=target_val)
report_test("Teste t (1 Amostra vs Meta 20 mpg)", t_1samp, stat_label="t")

```

#### 2. Teste $t$ de Welch (2 Amostras Independentes)
Usado quando as variâncias dos dois grupos podem ser diferentes (`equal_var=False`).
**Premissas**
- Distribuições normais nos grupos
- **NÃO** precisa de homocedasticidade. **Sempre**  use no lugar do Student

**Hipóteses**

- H_0: O x dos dois grupos tem mesma média
- H_0: O x dos dois grupos tem médias distintas

**O IC mostra quem tem a maior média!**: Se for valores positivos, o primeiro grupo é menor. Se negativo, maior.

**Plot os boxplots e veja qual a maior média**

**Conclusão**: *Em média o grupo a apresenta maior x que b*


```python
# Teste t de Welch
t_welch = stats.ttest_ind(group_manual, group_auto, equal_var=False)
report_test("Teste t de Welch (Manual vs Auto)", t_welch, stat_label="t")

# NOTE: Veja o IC!!. Se positivo, o primeiro grupo tem média menor

```

#### 3. Teste $t$ Pareado
Usado para medições antes/depois no mesmo indivíduo ou pares pareados.

**Premissas:**

- Pareados
- A diferença D tem distribuição aprox. normal

**Hipóteses:**
- H_0: A média das diferenças de performance de a e b é 0
- H_1: A média das diferenças é diferente de 0

**Exemplo:**
- H_0: Em média, a droga 2 não muda as horas de sono em relação a 1
- H_1: Em média, as horas de sono mudam entre as duas drogas

**Precisa fazer o pivot!**
```python
sleep_D = sleep.pivot(index="ID", columns=["group"], values="extra")
```

**O IC indica qual é o melhor!!. Aplica o teste t nas diferenças** 

```python
# Dados Simulados Pareados (ex: Horas de sono com Medicamento A vs Medicamento B nos mesmos pacientes)
droga_1 = np.array([0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0])
droga_2 = np.array([1.9,  0.8,  1.1,  0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4])

t_paired = stats.ttest_rel(droga_2, droga_1)
report_test("Teste t Pareado (Droga 2 vs Droga 1)", t_paired, stat_label="t")

# NOTE: Se IC é positivo, droga 2 é melhor, se não, droga 1

```

### 3.3 ANOVA One-Way e Post-Hoc Tukey HSD
Compara médias de 3 ou mais grupos independentes. Evita o acúmulo de erro Tipo I de múltiplos testes $t$.

**Premissas**
- Todos os grupos aprox. normais
- Homocedasticidade entre grupos (usar teste de Levene)

**Hipóteses**
- H_0: A média de x é igual nos n grupos
- H_0: A média de x é diferente em pelo menos um grupo

**Faça o teste Tukey depois para descobrir qual o diferente:**
- As classes são enumeradas pela ordem dos parâmetros
- Veja a célula *p-value*



```python
# Dados de Exemplo (Iris: Sepal Length por Espécie)
sp_setosa = np.array([5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9])
sp_versicolor = np.array([7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2])
sp_virginica = np.array([6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2])

# Executa a ANOVA
anova_res = stats.f_oneway(sp_setosa, sp_versicolor, sp_virginica)
report_test("ANOVA One-Way", anova_res, stat_label="F")

# Teste Post-Hoc Tukey HSD
tukey_res = stats.tukey_hsd(sp_setosa, sp_versicolor, sp_virginica)
print("\n--- Teste Post-Hoc Tukey HSD ---")
print(tukey_res)

```

### 3.4 Correlação de Pearson
Mede a associação linear entre duas variáveis contínuas.

**Vê se é RETA, o Spearman vê se há TENDÊNCIA**

**Premissas:**
- Relação linear (faça o scatterplot e veja se há curva)
- Normalidade bivariada

**Hipótese**
- H_0: Não existe relação linear entre as variáveis
- H_1: Existe relação linear entre as variáveis

**Se o scatterplot mostrar curva, tente fazer o log de uma delas**

```python
fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(mtcars, x="hp", y="mpg", ax=ax)
plt.show()

log_hp = np.log(mtcars["hp"])
```

#### Pearson básico
```python
# Exemplo: Relação entre Peso do Carro (wt) e Consumo (mpg)
wt = np.array([2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44])
mpg_corr = np.array([21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2])

pearson_res = stats.pearsonr(wt, mpg_corr)
report_test("Correlação de Pearson (wt vs mpg)", pearson_res, stat_label="r")

```

#### Pearson com gráfico e reta

```python
# Exemplo: Relação entre Peso do Carro (wt) e Consumo (mpg)
wt = np.array([2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44])
mpg_corr = np.array([21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2])

def interpretar_forca_pearson(teste_pearson):
    # Obtém o valor absoluto para avaliar a força independente da direção (positiva ou negativa)
    abs_r = abs(teste_pearson[0])
    
    if abs_r < 0.1:
        return "Inexistente ou muito fraca"
    elif abs_r < 0.3:
        return "Fraca"
    elif abs_r < 0.5:
        return "Moderada"
    elif abs_r < 0.7:
        return "Forte"
    else:
        return "Muito forte"
        
pearson_res = stats.pearsonr(mtcars["wt"], mtcars["mpg"])
report_test("Correlação de Pearson (wt vs mpg)", pearson_res, stat_label="r")

display(f"Força da correlação: {interpretar_forca_pearson(pearons_res)}")

# Dispersão dos valores e dos postos, com a reta de mínimos quadrados
fig, axes = plt.subplots( figsize=(8, 5))
x, y, xlabel, ylabel, name = mtcars["wt"], mtcars["mpg"], "Peso", "Consumo (mpg)", "Pearson"
slope, intercept = np.polyfit(x, y, deg=1)
grid = np.linspace(x.min(), x.max(), 100)
axes.scatter(x, y, color="#1F4E79", alpha=.8)
axes.plot(grid, intercept + slope * grid, color="#DD8452", linewidth=2)
axes.set(xlabel=xlabel, ylabel=ylabel,
        title=f"{name}: {stats.pearsonr(x, y).statistic:.2f}")
plt.tight_layout()
plt.show()
```

### 3.5 Teste $Z$ para Duas Proporções & Qui-Quadrado ($\chi^2$) de Independência

#### 1. Teste $Z$ de Duas Proporções
Compara taxas/proporções entre dois grupos independentes.

**Premissas**
- Variável binária
- Frequência mínima de 5 observacoes para 0 ou 1

**Hipóteses**
- H_0: A taxa de sucessos é igaul em ambos os grupos 
- H_1: Há diferença na taxa de sucessos

**Use o proportion_confint para pegar o intervalo de confiança e mostrar qual tem a taxa maior**




```python
# Exemplo: Sobrevivência por Sexo (ex: Titanic)
sucessos = np.array([233, 109])  # [Mulheres Sobreviventes, Homens Sobreviventes]
totais = np.array([314, 577])    # [Total Mulheres, Total Homens]

z_stat, p_z = proportions_ztest(sucessos, totais)
print(f" [Teste Z de Proporções] Z = {z_stat:.4f} · p = {p_z:.4e}")

ci_fem = proportion_confint(sucessos[0], totais[0])
ci_masc = proportion_confint(sucessos[1], totais[1])
print(f"   Taxa Mulheres: {sucessos[0]/totais[0]:.2%} | IC 95%: [{ci_fem[0]:.2%}, {ci_fem[1]:.2%}]")
print(f"   Taxa Homens:   {sucessos[1]/totais[1]:.2%} | IC 95%: [{ci_masc[0]:.2%}, {ci_masc[1]:.2%}]")

```

#### 2. Teste Qui-Quadrado ($\chi^2$) de Independência
Verifica se duas variáveis categóricas possuem associação.

**Premissas**
- Variáveis categóricas
- Faça a tabela de contingência (pd.crosstab). Deve haver mais de 5 observações em 80% das células

**Hipóteses**
- H_0:  As variáveis são independentes
- H_1:  As variáveis são dependentes

**Use os valores esperados sob H_0 para mostrar a frequência esperada e como cada grupo foge dela**
```python
# Tabela de Contingência: Classe da Passagem (1ª, 2ª, 3ª) vs Sobrevivência (0=Não, 1=Sim)
tabela_contingencia = np.array([
    [80, 136],   # 1ª Classe: [Faleceu, Sobreviveu]
    [97, 87],    # 2ª Classe
    [372, 119]   # 3ª Classe
])

# NOTE: Tabela
tab = pd.crosstab(df["pclass"], df["survived"])
chi2_stat, p_chi2, dof, expected = stats.chi2_contingency(tabela_contingencia)
print(f" [Qui-Quadrado de Independência] χ² = {chi2_stat:.4f} · gl = {dof} · p = {p_chi2:.4e}")
print("\nValores Esperados sob H0:\n", np.round(expected, 1))

```

## 4. Testes Não Paramétricos baseados em Postos (Aula 9)

Quando as premissas paramétricas (como normalidade ou ausência de outliers) falham, utilizamos **testes baseados em postos (ranks)**.

### 4.1 Cálculo de Postos (Ranks)



```python
dados_exemplo = np.array([1, 2, 2, 3, 5, 120])
postos = stats.rankdata(dados_exemplo)
print("Dados Originais:", dados_exemplo)
print("Postos (Ranks):", postos)

```

### 4.2 Teste de Mann-Whitney U & $\delta$ de Cliff
Substituto não paramétrico para o teste $t$ de duas amostras independentes. Avalia se uma distribuição tende a ter valores menores/maiores que a outra (dominância estocástica).

**Hipóteses**
- H_0: não há diferença na distribuição de x nos dois grupos
- H_1: Um grupo domina o outroH₀

**Exemplo do prof**
- H_0: Nenhum layout domina: P(A > B) = P(B > A).
- H_1: Um layout tende a gerar tempos menores que o outro.

**Se rejeitar H_0, usar o delta de cliff para achar o sentido e força**

```python
# Dados com Outlier Extremo (ex: Tempo de internação hospitalar em dias)
grupo_A = np.array([1, 2, 2, 3, 3, 4, 5, 120])  # Presença de outlier 120
grupo_B = np.array([4, 5, 5, 6, 6, 7, 8, 9])

mw_res = stats.mannwhitneyu(grupo_A, grupo_B, alternative="two-sided")
report_test("Mann-Whitney U", mw_res, stat_label="U")

# Tamanho do Efeito: Delta de Cliff
def cliffs_delta(g1, g2):
    n1, n2 = len(g1), len(g2)
    u_val, _ = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    delta = (2 * u_val - n1 * n2) / (n1 * n2)
    return delta

delta_c = cliffs_delta(grupo_A, grupo_B)
print(f"Delta de Cliff (δ): {delta_c:.4f} (Variando de -1 a +1)")
abs_delta = abs(delta)
if abs_delta < 0.147:
    forca = "Negligenciável"
elif abs_delta < 0.330:
    forca = "Pequena"
elif abs_delta < 0.474:
    forca = "Média"
else:
    forca = "Grande"
print(f"Força do cliff: {forca }")
print(f"Quem domina: {"grupo A" if delta_c > 0 else "grupo B" }")
```

### 4.3 Teste de Wilcoxon (Amostras Pareadas) & Kruskal-Wallis (ANOVA Não Paramétrica)


**Wilcoxon de uma amostra**

Testa se a mediana do gruop é igual ao alvo

**Se rejeitou, veja a mediana da amostra, se for maior, é maior que a meta**

```python
import numpy as np
from scipy import stats

# Dados do Grupo 1
grupo_1 = sleep[sleep['group'] == 1]['extra']

# Valor de referência (mediana teórica que queremos testar)
valor_referencia = 0.0

# Realizando o teste de uma amostra (passando a diferença)
estatistica, p_valor = stats.wilcoxon(grupo_1 - valor_referencia, alternative='two-sided')

print(f"Estatística W: {estatistica:.4f}")
print(f"p-valor: {p_valor:.4f}")
```


**Wilcoxon pareado**

H₀: A mediana das diferenças é zero. (MD = 0)
H₁: A mediana das diferenças não é zero. (MD ≠ 0)


**Kruskal-wallis**

**Hipóteses do prof**
- H_0: As três distribuições de notas são iguais.
- H_1: Pelo menos um algoritmo tende a receber notas maiores que outro.

**Se rejeitar, usa o dunn para formar a tabela e ver quais pares sao diferentes (ver se < 0,05)**

```python
# 1. Wilcoxon Signed-Rank Test (substituto do t pareado)
wilc_res = stats.wilcoxon(droga_2, droga_1, alternative="two-sided")
report_test("Wilcoxon Pareado", wilc_res, stat_label="W")

# 2. Kruskal-Wallis (substituto da ANOVA One-Way para 3+ grupos)
kruskal_res = stats.kruskal(sp_setosa, sp_versicolor, sp_virginica)
report_test("Kruskal-Wallis", kruskal_res, stat_label="H")

# 3. Se Kruskal rejeita H_0, usamos Dunn para encontrar
dunn_df = sp.posthoc_dunn([sp_setosa, sp_versicolor, sp_virginica],
p_adjust='bonferroni')


```

### 4.4 Correlação de Spearman
Mede a monotonicidade (se uma variável cresce quando a outra cresce, mesmo em curva).



```python
# Exemplo: Potência (hp) vs Consumo (mpg)
hp_vals = np.array([110, 110, 93, 110, 175, 105, 245, 62, 95, 123])
mpg_vals = np.array([21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2])

spearman_res = stats.spearmanr(hp_vals, mpg_vals)
report_test("Correlação de Spearman (Monotônica)", spearman_res, stat_label="ρ")

def interpretar_forca_spearman(rho):
    # Obtém o valor absoluto para avaliar a força independente da direção da correlação
    abs_rho = abs(rho)
    
    if abs_rho < 0.1:
        return "Inexistente ou muito fraca"
    elif abs_rho < 0.3:
        return "Fraca"
    elif abs_rho < 0.5:
        return "Moderada"
    elif abs_rho < 0.7:
        return "Forte"
    else:
        return "Muito forte"

# Exemplo de uso com o resultado obtido de Spearman no dataset cafe
rho_coef = spearman_res[0]
forca_spearman = interpretar_forca_spearman(rho_coef)
direcao_spearman = "negativa" if rho_coef < 0 else "positiva"

print(f"O coeficiente rho de Spearman é {rho_coef:.4f}")
print(f"Direção da associação: {direcao_spearman.upper()}")
print(f"Força da associação: {forca_spearman.upper()}")
```

### Cria matriz de correlação (pearson ou spearman)
```python
# Matrizes de Pearson e de Spearman entre todas as variáveis numéricas
numeric = mtcars.drop(columns="model")
mask = np.triu(np.ones((numeric.shape[1],) * 2, dtype=bool))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, method in zip(axes, ["pearson", "spearman"]):
    sns.heatmap(numeric.corr(method=method), mask=mask, ax=ax,
                cmap="vlag", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                square=True, cbar_kws={"shrink": .7})
    ax.set_title(method.capitalize())

plt.tight_layout()
plt.show()

## 5. Regressão Linear & Aprendizado de Máquina (Aula 10)

### 5.1 Protocolo de Divisão Treino / Teste Cego
Evita o vazamento de dados e avalia a capacidade de generalização real do modelo.
```



```python
# Dataset Sintético de Exemplo (Regressão Múltipla)
np.random.seed(42)
n_samples = 200

X_df = pd.DataFrame({
    "MedInc": np.random.uniform(1.5, 8.5, n_samples),
    "HouseAge": np.random.uniform(5, 50, n_samples),
    "AveRooms": np.random.uniform(3, 8, n_samples),
    "Population": np.random.uniform(300, 3000, n_samples)
})

# Resposta verdadeira com ruído
y_vec = 0.5 + 0.4 * X_df["MedInc"] + 0.01 * X_df["HouseAge"] - 0.05 * X_df["AveRooms"] + np.random.normal(0, 0.5, n_samples)

# Divisão 80% Treino / 20% Teste Cego
X_train, X_test, y_train, y_test = train_test_split(X_df, y_vec, test_size=0.20, random_state=42)

print(f"📊 Dataset Total: {n_samples} | Treino: {len(X_train)} | Teste Cego: {len(X_test)}")

```

### 5.2 Regressão Linear Múltipla & Métricas de Desempenho ($MAE$, $RMSE$, $R^2$)



```python
# Ajuste do Modelo nos dados de Treino
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Impressão dos Coeficientes
coef_df = pd.DataFrame({
    "Atributo": X_train.columns,
    "Coeficiente (β)": model_lr.coef_
})
print(f"Intercepto (β0): {model_lr.intercept_:.4f}\n")
print(coef_df.to_string(index=False))

# Avaliação no Treino vs Referência (Dummy)
dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_train)
y_pred_train = model_lr.predict(X_train)

def get_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred)
    }

metrics_summary = pd.DataFrame({
    "Modelo Referência (Média)": get_metrics(y_train, y_pred_dummy),
    "Regressão Linear (Treino)": get_metrics(y_train, y_pred_train)
}).T

print("\n--- Métricas no Conjunto de Treino ---")
print(metrics_summary.round(4))

```

### 5.3 Validação Cruzada $K$-Fold ($5$ Dobras)
Evita o viés do erro de treino e estima com precisão o desempenho em dados não vistos sem abrir o Teste Cego.



```python
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scoring_dict = {
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2"
}

cv_results = cross_validate(
    LinearRegression(), X_train, y_train,
    cv=kfold, scoring=scoring_dict, return_train_score=True
)

cv_df = pd.DataFrame({
    "R² Treino": cv_results["train_r2"],
    "R² Validação": cv_results["test_r2"],
    "RMSE Validação": -cv_results["test_rmse"]
}, index=pd.RangeIndex(1, 6, name="Dobra"))

print("--- Desempenho nas 5 Dobras de Validação Cruzada ---")
print(cv_df.round(4))

print("\nResumo da Validação Cruzada (Média ± Desvio Padrão):")
print(f"R² Validação:    {cv_df['R² Validação'].mean():.4f} ± {cv_df['R² Validação'].std():.4f}")
print(f"RMSE Validação:  {cv_df['RMSE Validação'].mean():.4f} ± {cv_df['RMSE Validação'].std():.4f}")

```

### 5.4 Diagnóstico por Gráficos de Resíduos
Avalia a homocedasticidade, linearidade e presença de padrões não capturados pelo modelo.



```python
# Previsões Out-of-Fold (Fora da Dobra)
oof_preds = cross_val_predict(LinearRegression(), X_train, y_train, cv=kfold)
residuos = y_train - oof_preds

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Painel 1: Valor Real vs Previsto
axes[0].scatter(y_train, oof_preds, alpha=0.6, color="#1F4E79")
min_val = min(y_train.min(), oof_preds.min())
max_val = max(y_train.max(), oof_preds.max())
axes[0].plot([min_val, max_val], [min_val, max_val], color="#DD8452", linestyle="--", label="Previsão Perfeita")
axes[0].set(xlabel="Valor Real (y)", ylabel="Valor Previsto (y_chapéu)", title="Valor Real vs Previsto (OOF)")
axes[0].legend()

# Painel 2: Resíduos vs Previsto
axes[1].scatter(oof_preds, residuos, alpha=0.6, color="#1F4E79")
axes[1].axhline(0, color="#DD8452", linestyle="--")
axes[1].set(xlabel="Valor Previsto (y_chapéu)", ylabel="Resíduo (y - y_chapéu)", title="Resíduos vs Previsto (Diagnóstico)")

plt.tight_layout()
plt.close()

```

### 5.5 Regularização: Ridge ($L_2$), Lasso ($L_1$) e ElasticNet com `Pipeline` e `StandardScaler`
A regularização adiciona uma penalidade ao tamanho dos coeficientes para controlar a variância, evitar o sobreajuste (*overfitting*) e tratar a multicolinearidade.

> **Importante**: Atributos devem ser padronizados antes da regularização! O uso de `make_pipeline` garante que a padronização seja aprendida apenas nas dobras de ajuste de cada rodada, evitando vazamento de dados (*data leakage*).



```python
# Definição das Pipelines para cada família de regressão
pipe_ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
pipe_lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.05))
pipe_elastic = make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.5))

modelos_dict = {
    "OLS (LinearSimples)": LinearRegression(),
    "Ridge (L2)": pipe_ridge,
    "Lasso (L1)": pipe_lasso,
    "ElasticNet": pipe_elastic
}

resultados_modelos = {}
for nome, mdl in modelos_dict.items():
    cv_res = cross_validate(mdl, X_train, y_train, cv=kfold, scoring=scoring_dict)
    resultados_modelos[nome] = {
        "R² (Validação CV)": cv_res["test_r2"].mean(),
        "RMSE (Validação CV)": -cv_res["test_rmse"].mean()
    }

print("--- Comparação de Modelos na Validação Cruzada ---")
print(pd.DataFrame(resultados_modelos).T.round(4))

```

### 5.6 Avaliação Final no Teste Cego
Apenas após todas as decisões de validação cruzada serem tomadas, o **Teste Cego** é aberto uma única vez para estimar a capacidade real de generalização.



```python
# Treino final no conjunto de Treino completo e previsão no Teste Cego
modelo_escolhido = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
modelo_escolhido.fit(X_train, y_train)

y_pred_test = modelo_escolhido.predict(X_test)
metrics_test = get_metrics(y_test, y_pred_test)

print("🏆 --- RESULTADO FINAL NO TESTE CEGO --- 🏆")
for metric_name, val in metrics_test.items():
    print(f"{metric_name:6s}: {val:.4f}")

```
