# Analise-Prescritiva-Amazon

Painel interativo em Streamlit para apoio à tomada de decisão logística e gestão de estoque a partir do **Amazon Delivery Dataset** (43.739 entregas em 44 dias de operação e 16 categorias de produtos).

## Estrutura do Repositório

- `app.py`: aplicação principal em Streamlit organizada em seis abas alinhadas às etapas do estudo de caso:
  1. Tratamento e Limpeza dos Dados (dicionário PT-BR / Inglês e tabelas tratadas)
  2. Regressão Linear e Diagramas de Dispersão (Demanda Diária x Tempo e Tempo de Entrega x Distância)
  3. Análise de Sensibilidade (Clima, Trânsito e Cenário Combinado)
  4. Otimização de Frota (tempo esperado ponderado pela frequência dos cenários)
  5. Ponto de Reposição e Simulação de Estoque (modelos de Reserva Percentual e Nível de Serviço Estatístico, simulação dos 44 dias, recomendação prescritiva consolidada e tabela de trade-off)
  6. Limitações, Premissas e Diagnósticos Estatísticos (CV%, sobredispersão, Shapiro-Wilk, autocorrelação e cobertura amostral)
- `amazon_pedidos_limpo.csv`: base tratada com 43.739 entregas.
- `amazon_estoque_diario.csv`: série diária agregada por produto (704 registros).
- `amazon_parametros_produtos.csv`: parâmetros consolidados das 16 categorias.
- `requirements.txt`: dependências Python (`streamlit`, `pandas`, `numpy`, `plotly`, `scipy`).

## Execução Local

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
