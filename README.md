RDN_Main.ipynb

Esse código é uma implementação básica de uma Rede Neural do zero para ajuste de uma curva linear usando Python e NumPy. 

1. **Imports e Configurações Gráficas**:
   - `import numpy as np`: Importa a biblioteca NumPy, usada para trabalhar com arrays e operações matemáticas.
   - `import matplotlib.pyplot as plt`: Importa a biblioteca Matplotlib para visualização gráfica.
   - `plt.style.use('dark_background')` e `plt.rcParams['figure.figsize'] = (10,8)`: Configuram o estilo e o tamanho dos gráficos, usando um fundo escuro para melhor contraste visual.

2. **Criação do Dataset**:
   - A função `get_linear_curve(x, w, b=0, noise_scale=0)` gera uma curva linear `y = wx + b` com ruído adicionado. A variável `noise_scale` controla a magnitude desse ruído.
   - `x = np.arange(-10, 30.1, 0.5)`: Cria uma sequência de valores entre -10 e 30, com incrementos de 0,5, representando a entrada `x` do dataset.
   - `Y = get_linear_curve(x, 1.8, 32, noise_scale=2.5)`: Gera os valores de `Y` baseados em uma inclinação `w = 1.8` e um intercepto `b = 32`, com ruído para simular variações reais.

3. **Visualização do Dataset**:
   - `plt.scatter(x, Y)`: Gera um gráfico de dispersão para visualizar o dataset gerado.
   - `plt.xlabel('°C', fontsize=20)` e `plt.ylabel('°F', fontsize=20)`: Definem os rótulos dos eixos X (temperatura em °C) e Y (temperatura em °F).

4. **Inicialização de Pesos e Bias**:
   - `w = np.random.randn(1)` inicializa o peso `w` com um valor aleatório.
   - `b = 0` inicializa o bias como zero.

5. **Função Feedforward**:
   - `def forward(inputs, w, b)`: Esta função realiza a predição, calculando `y = w*x + b` para cada entrada.

6. **Cálculo da Perda**:
   - `def mse(Y, y)`: Implementa a função de perda Mean Squared Error (MSE), que calcula a média dos erros quadráticos entre as predições `y` e os valores reais `Y`.

7. **Backpropagation**:
   - `def backpropagation(inputs, outputs, targets, w, b, learning_rate)`: Esta função ajusta os pesos (`w`) e o bias (`b`) usando o gradiente descendente. 
   - Os gradientes `dw` e `db` são calculados a partir da derivada da função de perda em relação a `w` e `b`.

8. **Treinamento do Modelo**:
   - `def model_fit(inputs, targets, w, b, learning_rate=0.001, epochs=200)`: Esta função treina o modelo ajustando `w` e `b` em múltiplas épocas. A cada época, ela realiza a passagem forward, calcula a perda, e aplica o backpropagation.
   - A perda é impressa a cada 50 épocas para monitorar o progresso do treinamento.

9. **Execução do Treinamento**:
   - `x = np.arange(-10,10,2)` define novos valores de entrada.
   - `w, b = model_fit(x, Y, w, b, learning_rate=0.005, epochs=2000)`: O modelo é treinado por 2000 épocas, com uma taxa de aprendizado de 0.005.

10. **Visualização dos Resultados**:
   - `plt.scatter(x, Y)` exibe os pontos de dados reais.
   - `plt.plot(x, get_linear_curve(x, w, b), color='red', lw=3)` sobrepõe a linha ajustada pelo modelo em vermelho, mostrando o quanto a linha predita se ajusta aos dados reais.

### Comentários gerais:
- Este é um exemplo simples, mas eficaz, de como criar e treinar um modelo linear. 
- A estrutura é básica e não envolve bibliotecas específicas de Machine Learning como TensorFlow ou PyTorch, o que ajuda a entender os princípios fundamentais por trás do aprendizado de máquina.
