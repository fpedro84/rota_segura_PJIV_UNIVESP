# Documentação Técnica e Funcional — Rota Segura

**Projeto:** PI IV — UNIVESP  
**Nome da aplicação:** Rota Segura 
**Tipo:** Aplicação web de planejamento de rotas e análise de risco rodoviário  
**Versão:** V1.0.0

---

## 1. Visão geral

O Rota Segura é uma aplicação web que permite informar uma origem e um destino, consultar alternativas de rota e visualizar informações como distância, tempo estimado, índice de acidentes e risco previsto por um modelo de Machine Learning.

A solução utiliza serviços e bibliotecas gratuitos, sem exigir chaves de API na configuração descrita no projeto. A análise de acidentes utiliza dados abertos da Polícia Rodoviária Federal (PRF), enquanto o cálculo de rotas e a exibição do mapa utilizam serviços do ecossistema OpenStreetMap.

O projeto também inclui um painel analítico com filtros, indicadores, mapas e gráficos para apoiar a identificação de trechos rodoviários que merecem atenção.

## 2. Objetivos

### 2.1 Objetivo geral

Apoiar o planejamento de viagens rodoviárias por meio da consulta de rotas e da análise de dados históricos de acidentes, apresentando indicadores que auxiliem a compreensão do risco rodoviário.

### 2.2 Funcionalidades descritas

- Consultar rotas a partir de origem e destino.
- Apresentar até três alternativas de rota, conforme disponibilidade do serviço de roteamento.
- Exibir distância e tempo estimado de deslocamento.
- Calcular um índice relativo de acidentes para o corredor da rota.
- Estimar a probabilidade de um acidente ser grave, condicionada à ocorrência de acidente, por meio de Machine Learning.
- Consultar dados em um painel com filtros por região, UF, rodovia, período e horário.
- Exibir indicadores, mapas de trechos, distribuição por horário, evolução mensal e gravidade por rodovia.
- Identificar automaticamente trechos de atenção.
- Estimar o custo de pedágios federais cobertos pelo arquivo de dados configurado.

## 3. Tecnologias e fontes de dados

| Tecnologia ou fonte   | Finalidade descrita |
| Python                | Linguagem utilizada na aplicação e no processamento dos dados |
| Flask                 | Aplicação web, conforme a estrutura `app.py` e das rotas HTTP |
| OpenStreetMap         | Dados cartográficos e mapa-base |
| Nominatim             | Geocodificação de endereços ou locais |
| OSRM                  | Consulta e geração de alternativas de rota |
| Leaflet               | Exibição e interação com mapas na interface |
| Dados abertos da PRF  | Base histórica de acidentes rodoviários (`datatran*.csv`) |
| Dados da ANTT         |Localização das praças de pedagios e preços (`pedagios_federais_antt.csv`)|
| BallTree com distância haversine | Associação espacial entre a rota e os registros de acidentes |
| Regressão Logística, Random Forest e Gradient Boosting | Modelos comparados para estimar gravidade |
| CSS e templates HTML | Apresentação visual das páginas, incluindo `static/routesafe.css` e `templates/_sidebar.html` |

## 4. Estrutura e arquivos relevantes

A estrutura explicitamente mencionada na documentação de origem inclui:

projeto/
├── app.py
├── requirements.txt
├── .env.example
├── data/
│   ├── datatran*.csv
│   └── pedagios_federais_antt.csv
├── .cache/
├── static/
│   └── routesafe.css
└── templates/  
     └── _sidebar.html
     └── landing.html
     └── painel.html
     └── rota.html

### 4.1 Arquivos e diretórios

- `app.py`: ponto de entrada usado para iniciar a aplicação web.
- `requirements.txt`: lista de dependências Python instaladas pelo `pip`.
- `.env.example`: exemplo de configurações opcionais.
- `data/`: diretório destinado aos CSVs de acidentes e, opcionalmente, de pedágios.
- `.cache/`: diretório onde são armazenados resultados de processamento e comparação dos modelos.
- `static/routesafe.css`: folha de estilos compartilhada pelas páginas.
- `templates/_sidebar.html`: componente compartilhado de navegação lateral.

## 5. Instalação e execução

### 5.1 Pré-requisitos

- Python instalado.
- Arquivos de dados da PRF colocados no diretório `data/`. Os arquivos podem ser obtidos em https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf
    -Documento CSV de Acidentes (ANO) (Agrupados por ocorrência) - Baixar planilha
- Acesso à internet para utilizar os serviços públicos de geocodificação, roteamento e mapas, salvo quando houver serviços substitutos configurados.
- Dependências listadas em `requirements.txt`.

### 5.2 Criar e ativar o ambiente virtual

No Windows, usando o terminal na pasta do projeto:

```bash
python -m venv .venv
.venv\Scripts\activate
```

No Linux ou macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 5.3 Instalar dependências

```bash
python -m pip install -r requirements.txt
```

### 5.4 Preparar os dados

Copie os arquivos da PRF obtidos em https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf Documento CSV de Acidentes *ANO (Agrupados por ocorrência) - Baixar planilha com o padrão `datatran*.csv` para o diretório `data/`. Vários anos podem ser concatenados por exemplo o teste foi realizado com dados de 2021 a 2026, totalizando aproximadamente 391 mil acidentes no ambiente descrito.

Para habilitar a estimativa de pedágios, coloque também `pedagios_federais_antt.csv` em `data/`, seguindo o formato especificado na seção 9.

### 5.5 Iniciar a aplicação

```bash
python app.py
```

Acesse no navegador:

```text
http://localhost:5000
```

### 5.6 Primeiro processamento e cache

Na primeira inicialização a aplicação compara três modelos de Machine Learning. Esse processamento pode levar aproximadamente 40 segundos a 2 minutos, dependendo do computador. Os resultados são guardados em `.cache/`, reduzindo o tempo das inicializações seguintes para poucos segundos.

O cache é reconstruído automaticamente quando os CSVs mudam.

## 6. Configuração opcional

É possível copiar `.env.example` para `.env` e ajustar as configurações abaixo:

| Variável          | Função descrita |
| `OSRM_URL`        | Define o endereço do serviço OSRM |
| `NOMINATIM_EMAIL` | Configuração de identificação para consultas ao Nominatim |
| `RAIO_CORREDOR_M` | Define o raio do corredor utilizado para relacionar acidentes à rota |

## 7. Módulos e telas

### 7.1 Planejamento de rota — `/`

Tela principal para informar origem e destino. A aplicação consulta alternativas de trajeto e apresenta distância, tempo estimado, índice de acidentes e risco previsto.

A disponibilidade e a diversidade das alternativas dependem do serviço de roteamento. O tempo apresentado é uma estimativa sem consideração de trânsito em tempo real.

### 7.2 Painel de Análise de Risco Rodoviário — `/painel`

Painel analítico acessível pelo botão na parte superior da tela inicial, oferece:

- Filtros por região, UF, rodovia, período e horário.
- Indicadores-chave (KPIs).
- Mapa de trechos.
- Análise por horário.
- Evolução mensal.
- Gravidade por rodovia.
- Detecção automática de trechos de atenção.

O painel utiliza os mesmos dados e o mesmo modelo disponíveis no servidor.

## 8. Metodologia de análise de acidentes

### 8.1 Preparação e localização dos acidentes

A quilometragem informada pela PRF reinicia por unidade federativa. Por esse motivo, a identificação de um trecho considera conjuntamente **BR + UF + km**, evitando tratar o mesmo quilômetro de rodovias em estados diferentes como um único local.

A geometria da rota é densificada com pontos a cada aproximadamente 60 metros. Em seguida, os pontos da rota são comparados espacialmente com os registros de acidentes dentro de um corredor configurado por `RAIO_CORREDOR_M`, cujo valor é de 150 metros. A busca espacial utiliza BallTree com distância haversine.

Essa associação é geométrica: um acidente próximo da linha da rota pode pertencer a outra via que cruza ou acompanha o trajeto.

### 8.2 Índice relativo de acidentes

O uso de Unidades Padrão de Severidade (UPS), atribuídas da seguinte forma:

| Gravidade registrada  | Peso UPS  |
| Sem vítimas           | 1         |
| Com feridos           | 4         |
| Com mortos            | 13        |

O índice compara a UPS por quilômetro da rota com a UPS por quilômetro média das rodovias federais da base:

**Índice = (UPS por km da rota) / (UPS por km médio das BRs na base)**

A classificação informada é:

| Classificação | Regra descrita                                |
| Baixo         | Índice menor ou igual a 0,8                   |
| Médio         | Índice maior que 0,8 e menor ou igual a 1,5   |
| Alto          | Índice maior que 1,5                          |

## 9. Machine Learning: estimativa de gravidade

A aplicação compara três algoritmos:

1. Regressão Logística.
2. Random Forest.
3. Gradient Boosting.

A avaliação descrita utiliza uma divisão temporal de 80% para treino e 20% para teste e considera a métrica AUC. O melhor modelo é utilizado para estimar a probabilidade de um acidente ser grave entre os acidentes associados ao corredor da rota, considerando o dia e o horário de saída.

### 9.1 Interpretação correta da previsão

A previsão **não representa a probabilidade de ocorrer um acidente durante a viagem**. Ela estima a gravidade, condicionada à ocorrência de um acidente, com base nos dados e nas variáveis utilizadas pelo modelo.

## 10. Estimativa de pedágios — ANTT

A aplicação pode estimar o custo dos pedágios federais presentes no arquivo `data/pedagios_federais_antt.csv`.

### 10.1 Formato do arquivo

text
id,nome,rodovia,uf,latitude,longitude,categoria,preco

### 10.2 Como a estimativa funciona

- O arquivo é carregado durante a inicialização.
- Uma praça é considerada quando está a até `PEDAGIO_RAIO_M` do trajeto.
- O raio informado é de 500 metros.
- O total soma o preço da categoria selecionada (moto, carro ou caminhões) por passagem, na ordem do percurso.

### 10.3 Cobertura e limitações

O arquivo descrito cobre somente os estados SP, RJ, MG, PR, ES, GO, SC, RS e MS que estão registrados na base de dados da ANTT. Praças de outros estados e pedágios estaduais não entram na estimativa. Se o arquivo não estiver disponível, o recurso é desativado.

## 11. Serviços externos e limites de uso

### 11.1 Nominatim

Utilizado para geocodificação, limite de até uma consulta por segundo, a aplicação aguarda um segundo entre buscas e mantém resultados em cache.

### 11.2 OSRM público

O serviço público `router.project-osrm.org` é indicado para testes e uso leve, sem garantia de disponibilidade.

### 11.3 OpenStreetMap e Leaflet

O OpenStreetMap fornece o mapa-base e o Leaflet é utilizado para apresentar o mapa na interface. A disponibilidade dos serviços externos e dos tiles pode afetar a experiência da aplicação.

### 11.4 Tempo de deslocamento

O tempo fornecido é uma estimativa de roteamento e **não considera o trânsito em tempo real**.

## 12. Limitações conhecidas

As seguintes limitações devem ser explicitadas ao apresentar os resultados:

1. **Gravidade, não ocorrência:** o modelo estima a gravidade dado que houve acidente; não estima a probabilidade de acontecer um acidente. A base descrita não inclui exposição ou fluxo de veículos.
2. **Cobertura da PRF:** os registros se referem a rodovias federais e rodovias conveniadas cobertas pela base, não a toda a malha viária.
3. **Correspondência geométrica:** o cruzamento espacial pode associar à rota acidentes de outra via próxima.
4. **Alternativas de rota:** o OSRM pode retornar poucas alternativas ou trajetos iguais, especialmente em viagens longas.
5. **Variáveis não utilizadas:** tipo e causa do acidente não são usados pelo modelo, pois não são conhecidos antes da viagem.
6. **Pedágios incompletos:** a cobertura do CSV limita a estimativa aos estados e praças presentes no arquivo.
7. **Dependência de serviços externos:** os serviços públicos podem ter limites, indisponibilidade ou alterações.

Os índices e previsões devem ser interpretados como indicadores de apoio à análise e ao planejamento, não como garantia de segurança ou substitutos de decisões responsáveis durante a viagem.
