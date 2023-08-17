# Autores:Bruno Machado Ferreira, Ernani Neto, Fábio Gomes e Ryan Nantes
## Filtro de Partículas

Testado com
- Conda 4.12.0
- Python 3.7.13
- Scikit-Image 0.16 e 0.19.2
- Numpy 1.21.5
- OpenCV 4.1.0
- OpenCV-Python 4.5.5.64
- Scipy 1.7.3
- Ubuntu 24.04 LTS

### COMANDOS NECESSÁRIOS:
Criando Ambiente Virtual
- $ conda create -n vc
- $ conda env list
- $ conda activate vc
- $ conda install scikit-image opencv
- $ conda list
- 
Remove tudo para instalar de novo:
- $ conda remove --name vc --all

### Instalando dependências de:
Filtragem de partículas
- $ pip install pfilter      (Repositório utilizado)
- $ pip install particles    (Filtragem mais limpa)
- $ pip install filterpy     (Contém o filtro de Kalman e alguns filtros básicos)
- $ pip install pyfilter     (Contém algoritmos de reamostragem, modelos auxiliares e algoritmos avançados com base em PyTorch)
- $ pip install pykalman     (Mais implementações do filtro de Kalman)
- $ pip install simdkalman   (Filtro de Kalman básico implementado)	
- $ pip install torch-kalman (Implementações PyTorch)

Conda e Python
- $ conda install scikit-image=0.16 opencv scipy numpy   (Em caso de conflitos, usar o scikit 0.19.2)
- $ pip install opencv-python

### SITES CONSULTADOS:
Em maior parte, repositórios e funções de filtragem mencionados em meio aos códigos example_filter.py e e pfilter.py.

### EXPLICAÇÃO
**Desenvolvimento**: 
Usamos como base um código python(https://github.com/johnhw/pfilter) para nosso filtro. Ao comentar com o professor sobre isso, nos foi requisitado que implementássemos mais formatos para o filtro. Mesmo com duas semanas de pesquisa, não obtivemos sucesso com isso e tivemos que deixar apenas o filtro circular. O autor original afirma que é apenas um código para explicar o funcionamento de um Filtro de Partículas, sem muitas outras funções acrescentadas.

**Funciomanento**: 
O filtro usa uma quantidade determinada de partículas e ruídos para desenhar dentro do formato implementado de dimensões fornecidas no código. No nosso caso, um círculo. O objetivo é computar como a figura se move a partir de observações parciais com ruído. O peso das partículas é levado em consideração para prever as posições do círculo.
Para evitar que haja muita diferença entre os pesos e manter a precisão, as funções multinomial_resample e resample recolhem amostras a cada etapa para que os pesos se mantenham em valores próximos.

**O código**:
Implementa o uso repetido de uma imagem 32x32 para detectar o movimento do círculo 4D dentro da tela. Essa imagem é repetida no espectro com as medidas instruídas pela função blob (examplespfilter.py,)
As previsões do filtro são atualizadas com os dados fornecidos e passados para a função update(pfilter.py) e os pesos de cada partícula. Depois, insere a o blob 32x32 no espectro novamente, repetindo a cada observação.
A pasta "equacoes" contém uma imagem com as equações matemáticas usadas no filtro de partículas, obtida no repositório https://github.com/johnhw/pfilter
Separação de funções: Todos os alunos fizeram pesquisas e contribuiram para o funcionamento do código, sendo Bruno e Ernani os responsáveis pela parte técnica, garantindo que o código rodasse nas dependências informadas e Fábio e Ryan os encarregados pela pesquisa teórica para melhor entendermos o funcionamento de um filtro de partículas.

**Resultado esperado**:
Ao rodar o arquivo "example_filter.py" na pasta "examples", um círculo branco surgirá na tela rodeado por outros azuis e um verde com uma linha vermelha no centro.
O círculo branco é o "blob" que será rastreado pelos círculos azuis, as partículas.
A blob se move de maneira aleatória no espectro e será rastreada pelas partículas, que serão pesadas para prever a próxima posição do objeto. O círculo verde é o visual disso, sendo a linha vermelha no centro o vetor peso que apontará para a direção prevista. 
