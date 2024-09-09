<h1>RDN_Main</h1>

Esse código é uma implementação de um modelo básico de aprendizado de máquina que fiz a partir de uma aula do canal (https://www.youtube.com/@MLparahumanos), e está dividido em algumas partes importantes que vou deixar anotado de uma uma maneira mais simples e direta para facilitar meu estudo e aprendizado:

1. **Importando as bibliotecas**:
   - Aqui você está trazendo as bibliotecas utilizadas:
     - `numpy` para lidar com operações matemáticas e arrays.
     - `matplotlib` para gerar gráficos.
   - Aqui, eu defino que o gráfico terá um fundo escuro (`plt.style.use('dark_background'`), o que deixa a visualização mais agradável.

2. **Criando o dataset**:
   - A função `get_linear_curve` serve para gerar uma linha reta com base na fórmula `y = wx + b`. A diferença é que ela adiciona um pouco de "ruído" (aleatoriedade) aos dados, simulando algo mais próximo da realidade.
   - A variável `x` vai de -10 até 30 (com passo de 0.5), e o `Y` é gerado usando a função acima, com uma inclinação de `1.8` e um ponto inicial `b = 32`. Esse ruído (`noise_scale=2.5`) faz com que os pontos não fiquem exatamente na linha reta, imitando variações que encontramos no mundo real.

3. **Visualizando os dados**:
   - Com o `plt.scatter(x, Y)`, é possível desenhar um gráfico de dispersão dos dados gerados, que são as entradas `x` versus as saídas `Y`.
   - Aqui eu rotulo os eixos, indicando que estamos lidando com algo como graus Celsius (`°C`) e Fahrenheit (`°F`), sugerindo uma relação entre temperatura em diferentes escalas.

4. **Configurando o modelo**:
   - inicializa o peso (`w`) com um valor aleatório, e o `b` (bias) começa em zero.
   - A função `forward` realiza a predição, ou seja, calcula o valor de saída (`y`) a partir dos inputs, pesos e bias: `y = wx + b`.

5. **Calculando o erro**:
   - A função `mse` (Mean Squared Error) mede o quanto o modelo está errando, comparando as saídas preditas com as reais. Quanto menor esse valor, melhor o modelo está acertando.

6. **Ajustando os parâmetros (Backpropagation)**:
   - Aqui, o modelo vai "aprender" ajustando o peso `w` e o bias `b` de acordo com o erro. Se o erro for grande, ele faz um ajuste maior; se for pequeno, ajusta menos.
   - A ideia é que a cada iteração, o modelo vai ficando melhor.

7. **Treinamento**:
   - O `model_fit` é onde acontece o aprendizado de verdade. O modelo passa pelos dados várias vezes (no seu caso, 2000 épocas), ajustando os parâmetros a cada rodada.
   - Você também printa o erro de tempos em tempos (a cada 50 épocas), para ver se o modelo está melhorando.

8. **Exibindo o resultado**:
   - Após o treinamento, você desenha novamente os pontos reais e sobrepõe a linha ajustada (com o `plt.plot`) em vermelho, para ver visualmente se o modelo se aproximou bem dos dados.

### Resumindo:
Meu código está criando um modelo que tenta prever uma relação linear (como a conversão de Celsius para Fahrenheit), ajustando um peso e um bias para isso. Ele usa o básico de aprendizado de máquina para ajustar esses parâmetros de forma que a linha se aproxime o máximo possível dos dados reais. E, no final, conseguimos visualizar o quanto o modelo conseguiu se ajustar à realidade.

---

<h1>RDN_Ativação</h1>

Seguindo o modelo anterior eu fiz algumas mudanças e adicionei uma função de ativação e 


### Explicação do Código


1. **Função de Ativação**:
   - A função `activation` categoriza a saída linear em três classes: "Frio", "Agradável" e "Calor" com base em valores de temperatura.

2. **Feedforward**:
   - A função `forward` calcula a saída linear e aplica a função de ativação para obter categorias.


3. **Backpropagation**:
   - A função `backpropagation` ajusta os pesos e o bias com base no erro calculado, atualizando os parâmetros do modelo para minimizar a perda.

4. **Treinamento do Modelo**:
   - A função `model_fit` treina o modelo por um número definido de épocas, atualizando os parâmetros em cada iteração e imprimindo a perda a cada 50 épocas.

5. **Predição e Visualização**:
   - Após o treinamento, usamos o modelo para fazer predições e visualizamos os dados reais, as predições categorizadas e a linha de regressão.

6. **Exibição dos Resultados**:
    - Finalmente, exibimos as predições categorizadas.


<p>Observação em ambos os caso eu utilizei o https://colab.research.google.com/ para criar e executar meus codigos, recomendo que utilizem caso queiram testar</p>

