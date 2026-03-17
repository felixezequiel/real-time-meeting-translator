# PRD — Real-Time Meeting Translator

## Visao Geral

Aplicacao Windows que funciona como proxy de audio em tempo real, interceptando o audio do auto-falante (speaker) e do microfone para traduzir entre Portugues e Ingles com latencia maxima de 2-5 segundos. Totalmente local, sem dependencia de internet ou APIs pagas.

## Problema

Profissionais brasileiros que desejam trabalhar para o exterior enfrentam a barreira do idioma em reunioes online. Mesmo com conhecimento tecnico, a falta de fluencia no ingles impede a comunicacao eficiente em calls do dia a dia.

## Publico-Alvo

- Desenvolvedores brasileiros trabalhando ou buscando trabalho remoto internacional
- Profissionais que entendem ingles parcialmente mas nao conseguem acompanhar conversas em tempo real
- Amigos e colegas do autor na mesma situacao

## Solucao

Um aplicativo system tray para Windows que:

1. **Captura o audio do speaker** (o que os outros falam na call)
2. **Transcreve** o audio usando Speech-to-Text local (Whisper.cpp)
3. **Traduz** o texto para o idioma alvo usando modelo local (CTranslate2 + Opus-MT ou similar)
4. **Sintetiza** o texto traduzido em voz usando TTS local (Piper TTS)
5. **Injeta** o audio sintetizado de volta no dispositivo de audio virtual

O usuario da reuniao ouve a traducao no lugar do audio original. Quando o usuario fala em portugues, o sistema traduz para ingles e envia o audio sintetizado pelo microfone virtual.

---

## Requisitos Funcionais

### RF-01: Captura de Audio do Speaker
- Capturar audio do dispositivo de saida selecionado usando WASAPI Loopback
- Buffer de audio em chunks otimizados para o pipeline de STT
- Suporte a diferentes sample rates (16kHz, 44.1kHz, 48kHz) com resampling automatico

### RF-02: Captura de Audio do Microfone
- Capturar audio do microfone fisico do usuario
- Aplicar Voice Activity Detection (VAD) para enviar apenas quando ha fala
- Silky Fusion: suprimir audio original e substituir pelo traduzido

### RF-03: Speech-to-Text (STT)
- Transcricao local usando Whisper.cpp (modelo `base` ou `small` para baixa latencia)
- Processamento em chunks de 2-3 segundos para balancear qualidade vs latencia
- Suporte apenas a Portugues (pt) e Ingles (en) para minimizar uso de memoria

### RF-04: Traducao de Texto
- Traducao local usando CTranslate2 com modelos Opus-MT (Helsinki-NLP)
  - `opus-mt-en-pt` (Ingles -> Portugues)
  - `opus-mt-pt-en` (Portugues -> Ingles)
- Otimizacao para frases curtas e linguagem coloquial de reunioes

### RF-05: Text-to-Speech (TTS)
- Sintese de voz local usando Piper TTS
- Vozes pre-configuradas para PT-BR e EN-US
- Velocidade de fala ajustavel (padrao: 1.1x para compensar latencia)

### RF-06: Dispositivo de Audio Virtual
- Registrar dispositivos de audio virtuais no Windows para speaker e microfone
- Opcao 1 (recomendada para MVP): Usar VB-Cable como dependencia (instalacao automatizada)
- Opcao 2 (futuro): Driver proprio usando Windows Audio Device Driver (KMDF/UMDF)
- Rotear audio traduzido pelo dispositivo virtual para que apps de reuniao o consumam transparentemente

### RF-07: Interface do Usuario
- System tray icon com menu de contexto
- Painel flyout (estilo notificacao Windows) com:
  - **Direcao da traducao do Speaker**: dropdown (EN->PT ou PT->EN)
  - **Direcao da traducao do Microfone**: dropdown (PT->EN ou EN->PT)
  - **Dispositivo de entrada** (microfone fisico): dropdown
  - **Dispositivo de saida** (speaker fisico): dropdown
  - **Indicador de status**: ativo/inativo com toggle
  - **Indicador de latencia**: tempo medio do pipeline em ms
- Hotkey global para ativar/desativar (padrao: Ctrl+Shift+T)
- Estado persistido entre sessoes

### RF-08: Pipeline de Audio
- Fluxo completo do speaker:
  ```
  Speaker Audio -> WASAPI Loopback -> VAD -> STT (Whisper) -> Traducao -> TTS -> Audio Virtual Speaker
  ```
- Fluxo completo do microfone:
  ```
  Microfone Fisico -> VAD -> STT (Whisper) -> Traducao -> TTS -> Microfone Virtual
  ```
- Cada etapa do pipeline deve ser assicrona e non-blocking
- Buffer entre etapas para absorver picos de latencia

---

## Requisitos Nao-Funcionais

### RNF-01: Latencia
- **Target**: 2-3 segundos end-to-end
- **Maximo aceitavel**: 5 segundos
- Breakdown estimado do budget de latencia:
  | Etapa | Target | Max |
  |-------|--------|-----|
  | Captura + VAD | 100ms | 200ms |
  | STT (Whisper) | 800ms | 1500ms |
  | Traducao | 200ms | 500ms |
  | TTS | 400ms | 800ms |
  | Buffer + overhead | 500ms | 1000ms |
  | **Total** | **2000ms** | **4000ms** |

### RNF-02: Uso de Recursos
- RAM: maximo 1.5GB (incluindo modelos carregados)
- CPU: otimizado para rodar sem GPU dedicada (suporte a GPU como bonus)
- Modelos carregados em memoria ao iniciar, sem lazy loading durante uso

### RNF-03: Qualidade de Audio
- TTS deve ser inteligivel e natural o suficiente para uma conversa
- Sem artefatos de audio perceptiveis (clicks, pops, gaps)
- Transicao suave entre segmentos de audio traduzido

### RNF-04: Privacidade
- 100% local, nenhum dado sai da maquina
- Sem telemetria, sem analytics, sem logs remotos
- Audio processado nao e persistido em disco (apenas em memoria)

---

## Stack Tecnica

| Componente | Tecnologia | Justificativa |
|-----------|-----------|---------------|
| **Linguagem principal** | Rust | Performance, seguranca de memoria, excelente suporte a audio/FFI |
| **Audio capture** | WASAPI (via cpal/wasapi-rs) | API nativa do Windows, menor latencia possivel |
| **Audio virtual** | VB-Cable (MVP) | Gratuito, amplamente testado, sem necessidade de driver proprio |
| **STT** | whisper.cpp (via whisper-rs) | Melhor STT local, otimizado para CPU, modelos compactos |
| **VAD** | Silero VAD (via ONNX Runtime) | Leve, preciso, roda em CPU facilmente |
| **Traducao** | CTranslate2 + Opus-MT | Modelos compactos (<100MB por par), rapido em CPU |
| **TTS** | Piper TTS | Vozes naturais, leve, suporte a PT-BR e EN-US |
| **UI** | Windows native (win32/tray) | Minimalista, sem overhead de framework UI |
| **Config** | TOML | Simples, legivel, padrao no ecossistema Rust |

---

## Arquitetura de Alto Nivel

```
+------------------+     +------------------+
|  Speaker Fisico  |     | Microfone Fisico |
+--------+---------+     +--------+---------+
         |                         |
    WASAPI Loopback           WASAPI Capture
         |                         |
    +----v----+               +----v----+
    |   VAD   |               |   VAD   |
    +----+----+               +----+----+
         |                         |
    +----v--------+           +----v--------+
    | STT Whisper |           | STT Whisper |
    +----+--------+           +----+--------+
         |                         |
    +----v--------+           +----v--------+
    | Traducao    |           | Traducao    |
    | EN->PT      |           | PT->EN      |
    +----+--------+           +----+--------+
         |                         |
    +----v--------+           +----v--------+
    | TTS (PT-BR) |           | TTS (EN-US) |
    +----+--------+           +----+--------+
         |                         |
    +----v-----------+   +--------v---------+
    | Virtual Speaker|   | Virtual Microfone|
    | (VB-Cable Out) |   | (VB-Cable In)    |
    +----------------+   +------------------+
         |                         |
    Usuario ouve          App de reuniao
    em Portugues          recebe em Ingles
```

---

## MVP (v0.1)

### Escopo do MVP
1. Captura de audio do speaker via WASAPI Loopback
2. STT com Whisper.cpp (modelo `base.en` para ingles, `base` para portugues)
3. Traducao EN->PT com CTranslate2 + Opus-MT
4. TTS com Piper (voz PT-BR)
5. Output no VB-Cable (usuario precisa instalar separadamente)
6. System tray com toggle on/off e selecao de direcao
7. **Apenas direcao speaker** (o que os outros falam -> traducao para PT)

### Fora do MVP
- Traducao do microfone (v0.2)
- Instalador automatizado com VB-Cable bundled
- Driver de audio proprio
- Overlay com transcricao em texto
- Suporte a GPU
- Ajuste de velocidade do TTS
- Hotkeys customizaveis

---

## Metricas de Sucesso

| Metrica | Target MVP | Target v1.0 |
|---------|-----------|-------------|
| Latencia end-to-end | < 5s | < 3s |
| Acuracia STT (WER) | < 25% | < 15% |
| Qualidade traducao (BLEU) | > 30 | > 40 |
| Uso de RAM | < 2GB | < 1.5GB |
| Crash rate | < 1/hora | < 1/dia |

---

## Riscos e Mitigacoes

| Risco | Impacto | Mitigacao |
|-------|---------|-----------|
| Latencia do Whisper em CPU muito alta | Alto | Usar modelo `tiny` ou `base`, chunks menores, streaming parcial |
| Qualidade da traducao insuficiente para contexto tecnico | Medio | Glossario tecnico customizavel, fallback para transcricao sem traducao |
| VB-Cable descontinuado ou incompativel | Baixo | Migrar para driver proprio ou alternativa open-source |
| Audio do TTS nao natural o suficiente | Medio | Oferecer opcao de output apenas texto (overlay) como alternativa |
| Uso de memoria excede o aceitavel | Medio | Quantizacao dos modelos (int8), unload de modelos nao usados |

---

## Fases de Desenvolvimento

### Fase 1 — Fundacao (MVP)
- Setup do projeto Rust
- Pipeline de audio: captura WASAPI -> VAD -> buffer
- Integracao Whisper.cpp para STT
- Integracao CTranslate2 para traducao
- Integracao Piper para TTS
- Output via VB-Cable
- System tray basico

### Fase 2 — Bidirecional
- Pipeline de microfone (PT->EN)
- Dois pipelines paralelos independentes
- UI com selecao de dispositivos

### Fase 3 — Polish
- Instalador com setup automatizado
- Otimizacoes de latencia (streaming STT, cache de traducao)
- Hotkeys globais
- Persistencia de configuracao

### Fase 4 — Futuro
- Driver de audio virtual proprio
- Suporte a GPU (CUDA/DirectML)
- Overlay com transcricao em tempo real
- Mais pares de idioma sob demanda
