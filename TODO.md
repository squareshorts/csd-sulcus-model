# TODO

## Prioridade imediata

- Criar o ambiente Python do projeto.
- Instalar o pacote local com dependências de desenvolvimento e superfície.
- Rodar a suíte de testes para validar o estado real do repositório.
- Confirmar se as dependências opcionais de malha cortical (`nibabel`) estão funcionando.

## Ambiente e validação

- Executar `python -m pip install -e .[dev,surface]`.
- Executar `python -m pytest -q`.
- Registrar qualquer falha de teste e separar em:
  - problema de ambiente;
  - problema de dependência;
  - problema real de implementação.

## Lacunas em relação às orientações

- Adicionar explicitamente a 4ª família experimental completa no fluxo de superfície:
  - geometria apenas;
  - geometria + anisotropia;
  - geometria + vascular;
  - geometria + anisotropia + vascular.
- Atualizar `scripts/run_surface_representative.py` para incluir o caso `anisotropy=False` e `vascular_feedback=True`.
- Garantir que os outputs e resumos salvem as quatro famílias de forma padronizada.

## Dados de superfície

- Confirmar quais arquivos anatômicos reais estarão disponíveis:
  - `midthickness.surf.gii` ou `white` + `pial`;
  - `sulc`;
  - `thickness`;
  - `vascular-risk`, se houver.
- Rodar `scripts/prepare_surface_bundle.py` com dados reais.
- Validar o bundle gerado:
  - sem `NaN`;
  - campos no intervalo esperado;
  - `preferred_axis` com shape correta.

## Simulação de superfície

- Rodar `scripts/run_surface_representative.py --quick` no ambiente instalado.
- Rodar `scripts/run_surface_representative.py --mesh <bundle_real.npz>`.
- Verificar:
  - propagação finita;
  - velocidades finitas;
  - estados `F`, `C` e `O` estáveis;
  - saída gráfica e tabelas geradas.

## Métricas e análise

- Confirmar se já existem métricas suficientes para:
  - velocidade geodésica;
  - atraso entre eletrodos virtuais;
  - mínimo de `F`;
  - mínimo de `O`.
- Se faltar, criar script dedicado para extração padronizada dessas métricas em batch.
- Montar tabela-resumo comparando as 4 famílias.

## Organização de resultados

- Padronizar pasta para os novos resultados de superfície.
- Garantir presença de:
  - CSV de resumo;
  - JSON de resumo;
  - campos brutos por caso;
  - figura comparativa.

## Manuscrito

- Só atualizar o manuscrito depois de:
  - 1 caso representativo sólido;
  - 1 sweep curto completo;
  - figuras estáveis.
- Depois disso, alinhar o texto com a versão real implementada no código.
