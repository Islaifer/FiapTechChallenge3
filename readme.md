
# Tech Challenge 3

## Autores
- Adryen Simões de Oliveira

## Descrição

Este projeto representa o Tech Challenge 3 da FIAP para o curso de Machine Learning Engineering.

O objetivo do projeto é fazer uma análise de dados de voos de aviões dos EUA no período de 2015. Os dados se encontram em: https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj?usp=sharing

## Link do vídeo descritivo
https://youtu.be/rLtioxP_0Ug

## Gráficos da análise
caminho /Data/Plots

## Como usar
Para utilizar o projeto, será necessário ter o python e o node (com npm) instalado. Com isso, você poderá rodar o projeto localmente e acessar o dashboard via: http://localhost:4200, ou as apis no Host http://localhost:8000.

Para instalar e rodar localmente, vá para a sessão de instalação.

### Documentação da API

O projeto disponibiliza das seguintes rotas de api (Essas informações podem ser encontradas na rota /doc, onde é montado um swagger para as mesmas):

#### /v1/machineLearn/analysis
Verbo http: GET

Response Body (retorna 200):
```json
"string"
```
Descrição: Essa rota serve para começar a análise do projeto, já criando e salvando em memória os modelos de machine learn.

#### /v1/machineLearn/metrics
Verbo http: GET

Response Body (retorna 200):
```json
{
  "accuracy": 0,
  "precision": 0,
  "recall": 0,
  "f1": 0,
  "rmse": 0,
  "mae": 0,
  "silhouette": 0
}
```
Descrição: Essa rota serve para ver as métricas dos modelos após análise (retorna 0 para todos os valores caso a análise não tenha sido realizada).

#### /v1/machineLearn/regression/predict
Verbo http: POST

Request Body:
```json
{
  "distance": 0,
  "diverted": 0,
  "month": 0,
  "day": 0,
  "day_of_week": 0,
  "scheduled_departure_hour": 0,
  "origin": "string",
  "destination": "string",
  "airline": "string",
  "has_weather_delay": 0,
  "has_airline_delay": 0,
  "has_air_system_delay": 0,
  "has_security_delay": 0,
  "has_late_aircraft_delay": 0
}
```

Response Body:
```json
0
```
Descrição: Essa rota serve para fazer a predição no modelo de regressão (quanto tempo um atraso durará).

#### /v1/machineLearn/classifier/predict
Verbo http: POST

Request Body:
```json
{
  "distance": 0,
  "diverted": 0,
  "month": 0,
  "day": 0,
  "day_of_week": 0,
  "scheduled_departure_hour": 0,
  "origin": "string",
  "destination": "string",
  "airline": "string"
}
```

Response Body:
```json
0
```
Descrição: Essa rota serve para fazer a predição no modelo de classificação (se um voo atrasará ou não).

## Funcionamento

Este projeto é feito com FastAPI, e ele serve para disponibilizar as chamadas da análise, desde a realização da análise, até as predições.

Este projeto realiza as análises dos dados encontrados em: https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj?usp=sharing

Para seu funcionamento, baixe os dados dos aeroportos e voos e coloque na pasta Data/ do projeto, pois ele irá ler eles.

Após a leitura dos dados e a geração de features para melhor desempenho dos modelos, ele iniciará o treinamento de 3 modelos, um de classificação, um de regressão e um de clusterização.

Logo após, as métricas e predições estarão disponíveis pelas rotas. Essas rotas são usadas pelo FrontEnd, que mostra de maneira simples as análises do projeto.


### Escolhas do projeto
Foi escolhido o framework do FastApi pela sua velocidade e praticidade em disponibilizar endpoints.

Foi escolhido o modelo XGBClassifier e XGBRegressor pelo seu alto desempenho em muitos dados com muitas features, além dele ter um melhor desempenho comparado com outros modelos.

Foi escolhiso o modelo de KMeans por sua simplicidade em separar os dados.

## Instalação e Uso
Caso queira instalar e utilizar o projeto localmente, nessa sessão vamos ensinar a instalar e rodar o projeto.

### Requisitos
- python
- node v20

### Como instalar

Vá até o diretório que se encontra o projeto em sua máquina.
Baixe os arquivos de https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj?usp=sharing e mova o csv de aeroportos e voos para a pasta Data/ dentro do projeto.

Busque o repositório Server/
Crie um ambiente virtual e instale os requerimentos com:

```bash
  pip install -r requirements.txt 
``` 

Depois disso, rode o servidor com:

```bash
  uvicorn app.main:app
```

Com isso, o backend estará rodando e já poderá acessar as APIs no host http://localhost:8000. 
Para rodar o front end, vá na pasta Client/ do projeto e rode:

```bash
  npm install
```

E depois rode com:

```bash
  ng serve
```

Assim poderá acessar o dashboard iterativo em http://localhost:4200

