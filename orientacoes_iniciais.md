# Roteiro Operacional Inicial

Abaixo está um roteiro operacional, em português, para dois alunos executarem essas análises sem desviar para um pipeline pesado logo no início. A lógica é simples: um aluno foca em infraestrutura/superfície e o outro em análise/validação. Isso reduz retrabalho e acelera o caminho até um resultado publicável.

## Objetivo Geral

Implementar e testar uma versão em superfície cortical do modelo, mantendo a expansão restrita a:

- difusão/propagação de potássio na malha cortical;
- estados de recuperação e vasculares como ODEs por vértice;
- três variáveis vasculares apenas:
  - `F`: reserva de perfusão;
  - `C`: drive constritivo lento;
  - `O`: reserva metabólica/oxigenação;
- comparação de apenas quatro famílias de modelo:
  - geometria de superfície apenas;
  - geometria + anisotropia;
  - geometria + feedback vascular;
  - geometria + anisotropia + feedback vascular.

## Divisão de Trabalho Entre os Dois Alunos

### Aluno 1: infraestrutura e simulação em superfície

Fica responsável por:

- preparar a malha cortical;
- montar operadores de superfície;
- executar simulações representativas;
- garantir que o modelo roda de ponta a ponta.

### Aluno 2: análise quantitativa e validação

Fica responsável por:

- definir métricas;
- comparar variantes do modelo;
- gerar figuras e tabelas;
- organizar os resultados para manuscrito.

Essa divisão é a mais eficiente. Um constrói o motor; o outro mede se o motor presta.

## Plano de Execução por Etapas

### Etapa 1: Confirmar ambiente e estrutura do repositório

**Tarefa dos dois alunos**

- Abrir o repositório `C:\work\CSD`.
- Confirmar que os arquivos novos existem:
  - `src/csd_sulcus/surface_io.py`
  - `src/csd_sulcus/surface_ops.py`
  - `src/csd_sulcus/surface_model.py`
  - `src/csd_sulcus/surface_prep.py`
  - `scripts/run_surface_representative.py`
  - `scripts/prepare_surface_bundle.py`
  - `tests/test_surface_model.py`
- Instalar o pacote com extras de superfície:

```bash
python -m pip install -e .[surface]
```

- Rodar os testes:

```bash
pytest -q
```

- Verificar se tudo passa.

**Critério de sucesso**

O ambiente precisa estar funcional antes de qualquer análise. Se os testes falharem, não avancem.

### Etapa 2: Preparar os dados de superfície

**Aluno 1**

O foco aqui é preparar um bundle limpo para rodar o modelo.

**Caso A: vocês já têm malha GIFTI/HCP**

```bash
python scripts/prepare_surface_bundle.py \
  --mesh path\to\midthickness.surf.gii \
  --sulc path\to\sulc.shape.gii \
  --thickness path\to\thickness.shape.gii \
  --output data\lh_surface_bundle.npz
```

**Caso B: vocês têm saídas FreeSurfer**

```bash
python scripts/prepare_surface_bundle.py \
  --white path\to\lh.white \
  --pial path\to\lh.pial \
  --sulc path\to\lh.sulc \
  --thickness path\to\lh.thickness \
  --output data\lh_surface_bundle.npz
```

**Se o sinal do sulco estiver invertido**

Usar:

```bash
--sulc-sign positive-is-deep
```

Ou manter o padrão, se for convenção FreeSurfer:

```bash
negative-is-deep
```

**O que o Aluno 1 deve verificar**

- O bundle `.npz` foi criado.
- O bundle contém, no mínimo:
  - vértices;
  - faces;
  - sulcal depth normalizado;
  - thickness;
  - preferred-axis;
  - vascular-risk, se derivado automaticamente.
- Não há `NaN` nem valores absurdos.

**Saída esperada**

Arquivo tipo:

```text
data\lh_surface_bundle.npz
```

### Etapa 3: Rodar a simulação representativa

**Aluno 1**

Agora o objetivo é verificar se a branch de superfície funciona de ponta a ponta.

**Rodar teste rápido**

```bash
python scripts/run_surface_representative.py --quick
```

**Rodar com bundle real**

```bash
python scripts/run_surface_representative.py --mesh data\lh_surface_bundle.npz
```

**O que deve ser checado**

- O modelo roda sem travar.
- As velocidades de propagação são finitas.
- A frente de propagação se desloca sobre a superfície.
- Os estados `F`, `C` e `O` evoluem sem instabilidade numérica.

**Se algo der errado**

O Aluno 1 deve testar na seguinte ordem:

1. malha sintética;
2. bundle preparado;
3. bundle com campos reais;
4. anisotropia desligada;
5. vascular desligado.

Isso isola o problema rapidamente.

### Etapa 4: Validar os operadores de superfície

**Aluno 1**

Antes de fazer qualquer sweep, é obrigatório provar que a parte geométrica está minimamente correta.

**Procedimento**

- Rodar um caso simples de difusão pura ou tipo Barkley-like na malha.
- Comparar qualitativamente com o caso em folha plana.
- Verificar se o Laplaciano e a massa não geram explosão numérica.
- Testar:
  - isotrópico;
  - anisotrópico.
- Ver se a anisotropia altera direção/velocidade de propagação de forma plausível.

**Registro**

O Aluno 1 deve salvar:

- figura da malha;
- mapa espacial da variável de potássio;
- curva temporal média;
- print dos parâmetros usados.

Sem registro, depois ninguém reproduz nada.

### Etapa 5: Definir exatamente as quatro famílias de modelo

**Aluno 2**

O Aluno 2 deve organizar a matriz experimental. Nada de busca livre gigante.

**Família 1: Geometria de superfície apenas**

- superfície ligada;
- anisotropia desligada;
- vascular desligado.

**Família 2: Geometria + anisotropia**

- superfície ligada;
- anisotropia ligada;
- vascular desligado.

**Família 3: Geometria + vascular**

- superfície ligada;
- anisotropia desligada;
- vascular ligado.

**Família 4: Geometria + anisotropia + vascular**

- superfície ligada;
- anisotropia ligada;
- vascular ligado.

**Regra importante**

No primeiro artigo:

- usar uma malha atlas;
- um hemisfério só;
- um sítio de ignição representativo;
- sweep estreito de 2 a 3 parâmetros vasculares.

Não compliquem isso. O ganho marginal de expandir cedo é baixo e o custo explode.

### Etapa 6: Definir os parâmetros do sweep estreito

**Aluno 2**

O sweep inicial deve ser pequeno. O objetivo é gerar uma figura publicável, não mapear o universo.

**Sugestão de parâmetros para variar**

Escolher apenas 2 ou 3 entre:

- força do acoplamento de clearance com `F`;
- força do acoplamento de clearance com `O`;
- intensidade do drive constritivo `C`;
- limiar de excitabilidade modulado por vascular;
- intensidade da anisotropia.

**Exemplo de sweep racional**

- 2 níveis para anisotropia: baixo / alto;
- 2 níveis para feedback vascular: fraco / forte;
- 2 níveis para constrição lenta `C`: fraca / forte.

Isso já dá 8 casos. Está no intervalo aceitável.

**Limite recomendado**

- desenvolvimento: 8 casos;
- máximo para sweep curto: 16 casos.

Passou disso, vocês estão perdendo foco.

### Etapa 7: Rodar os casos de desenvolvimento

**Aluno 1 executa, Aluno 2 registra**

**Configuração recomendada**

- malha grossa ou reduzida;
- horizonte temporal curto;
- poucos casos;
- saída compacta.

**Objetivo**

Determinar:

- quais casos são estáveis;
- quais geram propagação clara;
- quais métricas já se separam entre grupos.

**Registro mínimo por caso**

- nome do caso;
- parâmetros;
- tempo de execução;
- sucesso/falha;
- velocidade média;
- atraso entre sítios;
- mínimo de `F`;
- mínimo de `O`.

O Aluno 2 deve montar uma planilha simples com isso.

### Etapa 8: Extrair as métricas principais

**Aluno 2**

Essa parte é crítica. Sem métrica objetiva, o resultado vira só animação bonita.

**Métrica 1: Velocidade geodésica de propagação**

Passos:

1. escolher um conjunto de vértices ou regiões-alvo;
2. calcular tempo de chegada da frente;
3. calcular distância geodésica ao sítio de ignição;
4. estimar velocidade = distância / atraso.

**Métrica 2: Delay entre "eletrodos" na malha**

Passos:

1. definir pares ou rede de vértices como eletrodos virtuais;
2. extrair o instante de cruzamento de limiar;
3. calcular atraso entre pares.

**Métrica 3: Mínimo de F**

Passos:

1. extrair a série temporal de `F` por vértice;
2. computar:
   - mínimo global;
   - mínimo na área recrutada;
   - tempo até o mínimo.

**Métrica 4: Mínimo de O**

Mesmo procedimento de `F`.

**Métrica 5: Decomposição transporte versus vascular**

Aqui o Aluno 2 deve separar, de modo operacional, o quanto da alteração vem de:

- geometria/aniso de transporte;
- feedback vascular.

**Como fazer isso sem inventar moda**

Comparar diferenças entre famílias:

- Família 2 menos Família 1 = contribuição do transporte anisotrópico;
- Família 3 menos Família 1 = contribuição vascular;
- Família 4 menos Família 2 = vascular sobre transporte anisotrópico;
- Família 4 menos Família 3 = anisotropia sobre contexto vascular.

Isso já é suficiente para uma decomposição interpretável.

### Etapa 9: Organizar saídas gráficas

**Aluno 2**

Gerar um conjunto pequeno e sólido de figuras.

**Figura 1**

Malha cortical com sítio de ignição e direção preferencial de transporte.

**Figura 2**

Mapas de propagação no tempo para as 4 famílias.

**Figura 3**

Velocidade geodésica por família.

**Figura 4**

Atraso entre eletrodos virtuais por família.

**Figura 5**

Mínimos de `F` e `O`.

**Figura 6**

Gráfico de decomposição: transporte vs vascular.

**Regra**

Menos figuras, melhor. Cada figura precisa responder uma pergunta.

### Etapa 10: Rodar o batch final

**Aluno 1**

Depois do desenvolvimento:

- fazer o batch final com:
  - malha média resolução;
  - horizonte temporal suficiente;
  - casos finais definidos;
  - saída salva de modo padronizado.

**Ideal**

Rodar overnight localmente.

**O que salvar**

- parâmetros;
- seeds, se houver;
- outputs brutos;
- métricas resumidas;
- figuras.

Criem diretórios organizados, por exemplo:

```text
results/
  representative/
  sweep/
  figures/
  tables/
  logs/
```

### Etapa 11: Tabela final para manuscrito

**Aluno 2**

Criar uma tabela com colunas como:

- família do modelo;
- anisotropia;
- vascular;
- parâmetro vascular 1;
- parâmetro vascular 2;
- velocidade geodésica;
- atraso médio;
- mínimo de `F`;
- mínimo de `O`;
- observação qualitativa.

Essa tabela é a ponte direta para o texto do artigo.

### Etapa 12: Atualizar o manuscrito só depois dos resultados

**Ambos**

Só depois que houver pelo menos:

- 1 caso representativo sólido;
- 1 sweep pequeno completo;
- 1 conjunto de figuras estáveis.

Aí sim atualizar:

```text
manuscript/reframed_submission.tex
```

**O que inserir primeiro**

- descrição do modelo em superfície;
- descrição do acoplamento vascular mínimo;
- desenho experimental das 4 famílias;
- métricas;
- principais resultados.

Não escrevam método especulativo antes do modelo estar funcionando de verdade.

## Cronograma Prático de 5 Dias

### Dia 1

**Aluno 1**

- preparar bundle de superfície;
- testar `run_surface_representative.py`;
- validar se a malha real carrega.

**Aluno 2**

- definir planilha de casos;
- definir métricas;
- definir eletrodos virtuais e sítio de ignição.

### Dia 2

**Aluno 1**

- validar geometria e difusão em superfície;
- testar isotropia vs anisotropia.

**Aluno 2**

- escrever scripts de extração de velocidade e atraso;
- desenhar template das figuras.

### Dia 3

**Aluno 1**

- ligar `F`, `C`, `O`;
- testar estabilidade numérica;
- rodar 2 a 4 casos piloto.

**Aluno 2**

- extrair `Fmin`, `Omin`;
- comparar famílias preliminares.

### Dia 4

**Aluno 1**

- rodar 8 a 16 casos curtos.

**Aluno 2**

- consolidar métricas;
- gerar figuras comparativas;
- identificar os casos finais.

### Dia 5

**Aluno 1**

- rodar os casos finais em resolução média.

**Aluno 2**

- fechar tabelas;
- fechar figuras;
- redigir resultados preliminares.

## Checklist Objetivo para os Dois Alunos

### Checklist do Aluno 1

- ambiente instalado;
- testes passaram;
- bundle de superfície criado;
- simulação rápida rodou;
- simulação com bundle real rodou;
- isotropia/aniso testadas;
- vascular testado;
- batch final executado;
- outputs salvos e organizados.

### Checklist do Aluno 2

- matriz experimental definida;
- métricas implementadas;
- eletrodos virtuais definidos;
- tabela-resumo pronta;
- figuras principais prontas;
- comparação entre 4 famílias concluída;
- resultados preparados para manuscrito.

## O Que Não Fazer

Não façam isso agora:

- pipeline subject-specific completo no primeiro ciclo;
- sweep grande de parâmetros;
- múltiplos hemisférios;
- muitos sítios de ignição;
- acoplamento biofísico detalhado além de `F`, `C`, `O`;
- reescrever o artigo inteiro antes dos resultados.

Isso só atrasa.

## Resultado Mínimo Publicável Esperado

Ao final, vocês devem ter:

- uma simulação representativa em superfície cortical;
- comparação das 4 famílias do modelo;
- evidência de efeito de anisotropia e/ou vascularização em:
  - velocidade;
  - atraso;
  - reserva perfusional/metabólica;
- 1 tabela e 4 a 6 figuras limpas;
- texto preliminar de métodos e resultados.

Isso já configura um caminho de manuscrito plausível.

Se quiser, eu posso converter isso agora em um protocolo formal de laboratório, em português, com tarefas diárias separadas por aluno e caixas de verificação.
