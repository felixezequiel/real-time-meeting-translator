# Implementation Plan — MVP (Fase 1)

## Escopo

Tradutor de audio em tempo real para reunioes, direcao speaker apenas (EN->PT).
Pipeline: WASAPI Loopback -> VAD -> Whisper STT -> Opus-MT Translation -> Piper TTS -> VB-Cable Output.
System tray com toggle on/off e selecao de direcao.

## ADRs Relacionados

Nenhum ADR criado ainda. ADRs serao criados conforme decisoes arquiteturais surgirem durante a implementacao.

## Mapa de Dependencias

```
T-01 (Scaffolding)
  |
  +---> T-02 (Audio Capture WASAPI)
  |       |
  |       +---> T-03 (VAD - Silero)
  |               |
  |               +---> T-05 (Pipeline Orchestrator)
  |                       |
  +---> T-04 (Audio Output VB-Cable)  ---> T-05
  |                                         |
  +---> T-06 (STT - Whisper)  -----------> T-08 (Integration)
  |                                         |
  +---> T-07 (Translation - Opus-MT) ----> T-08
  |                                         |
  +---> T-09 (TTS - Piper)  ------------> T-08
  |                                         |
  +---> T-10 (System Tray UI) -----------> T-11 (End-to-End)
  |                                         |
  T-08 (Integration) --------------------> T-11
```

## Tasks

---

### T-01: Project Scaffolding + Domain Types

**Objetivo:** Criar o projeto Rust com workspace, dependencias e tipos do dominio que todas as tasks usarao.

**Escopo:**
- `cargo init` com workspace structure
- Definir crates: `audio`, `stt`, `translation`, `tts`, `pipeline`, `ui`
- Tipos do dominio compartilhados:
  - `AudioChunk` (samples: Vec<f32>, sample_rate: u32, channels: u16)
  - `TextSegment` (text: String, language: Language)
  - `Language` enum (Portuguese, English)
  - `TranslationDirection` (source: Language, target: Language)
  - `PipelineStage` trait (process async)
  - `PipelineConfig` struct
- `Cargo.toml` com todas as dependencias declaradas
- Config TOML basico com defaults

**Criterios de Aceite:**
- [x] `cargo build` compila sem erros
- [x] Tipos do dominio definidos com testes unitarios
- [x] Workspace com crates separados compila
- [x] Config TOML carrega com defaults

**Complexidade: Baixa**
| Eixo | Nivel |
|------|-------|
| Conceitos simultaneos | 2 (workspace layout + domain types) |
| Decisoes nao-obvias | 1 (granularidade dos crates) |
| Fronteiras cruzadas | 1 (shared types) |
| Comportamento alterado | Novo codigo |

---

### T-02: Audio Capture — WASAPI Loopback

**Objetivo:** Capturar audio do speaker do sistema via WASAPI Loopback e emitir AudioChunks.

**Depende de:** T-01

**Escopo:**
- Listar dispositivos de saida de audio do Windows
- Captura via WASAPI Loopback no dispositivo selecionado
- Resampling para 16kHz mono (formato esperado pelo Whisper)
- Emitir `AudioChunk` via channel async (tokio::sync::mpsc)
- Chunking configuravel (padrao: 2 segundos de audio)

**Criterios de Aceite:**
- [ ] Lista dispositivos de audio do sistema
- [ ] Captura audio do speaker default em loopback
- [ ] Resampling para 16kHz mono funciona
- [ ] AudioChunks emitidos via channel com tamanho configuravel
- [ ] Teste: captura 5 segundos de audio e salva em WAV para validacao manual

**Complexidade: Media**
| Eixo | Nivel |
|------|-------|
| Conceitos simultaneos | 3 (WASAPI, resampling, async channels) |
| Decisoes nao-obvias | 2 (buffer size, resampling strategy) |
| Fronteiras cruzadas | 1 (audio crate) |
| Comportamento alterado | Novo codigo |

---

### T-03: Voice Activity Detection — Silero VAD

**Objetivo:** Filtrar AudioChunks para processar apenas segmentos com fala, economizando CPU.

**Depende de:** T-02

**Escopo:**
- Integrar Silero VAD via ONNX Runtime
- Receber `AudioChunk` do channel de captura
- Classificar como fala/silencio
- Emitir apenas chunks com fala para o proximo estagio
- Threshold configuravel de sensibilidade

**Criterios de Aceite:**
- [ ] Modelo Silero VAD carrega via ONNX Runtime
- [ ] Detecta fala vs silencio com acuracia > 90%
- [ ] Chunks de silencio sao descartados (nao seguem no pipeline)
- [ ] Latencia do VAD < 50ms por chunk
- [ ] Teste: gravar audio com pausas, validar que apenas fala passa

**Complexidade: Baixa**
| Eixo | Nivel |
|------|-------|
| Conceitos simultaneos | 2 (ONNX inference, audio classification) |
| Decisoes nao-obvias | 1 (threshold tuning) |
| Fronteiras cruzadas | 1 (audio crate) |
| Comportamento alterado | Novo codigo |

---

### T-04: Audio Output — VB-Cable Playback

**Objetivo:** Reproduzir audio sintetizado no dispositivo VB-Cable para que o usuario ouca a traducao.

**Depende de:** T-01

**Escopo:**
- Listar dispositivos de saida e identificar VB-Cable
- Playback de `AudioChunk` via WASAPI no dispositivo VB-Cable
- Buffer de playback para transicao suave entre chunks
- Crossfade entre segmentos para evitar clicks/pops

**Criterios de Aceite:**
- [ ] Detecta VB-Cable entre os dispositivos disponiveis
- [ ] Reproduz AudioChunk no VB-Cable sem artefatos
- [ ] Buffer absorve variacoes de timing do pipeline
- [ ] Teste: reproduzir arquivo WAV no VB-Cable, ouvir no dispositivo configurado

**Complexidade: Media**
| Eixo | Nivel |
|------|-------|
| Conceitos simultaneos | 3 (WASAPI render, buffering, crossfade) |
| Decisoes nao-obvias | 2 (buffer size, crossfade strategy) |
| Fronteiras cruzadas | 1 (audio crate) |
| Comportamento alterado | Novo codigo |

---

### T-05: Pipeline Orchestrator — Async Stage Coordination

**Objetivo:** Orquestrar os estagios do pipeline com channels async, conectando captura -> VAD -> STT -> traducao -> TTS -> output.

**Depende de:** T-02, T-03, T-04

**Escopo:**
- Pipeline runner que conecta stages via mpsc channels
- Cada stage roda em sua propria tokio task
- Start/stop do pipeline via comando
- Metricas de latencia por stage (timestamp em cada handoff)
- Error handling: stage falhou -> log + skip chunk (nao para o pipeline)

**Criterios de Aceite:**
- [ ] Pipeline conecta N stages via channels
- [ ] Cada stage roda em task independente
- [ ] Start/stop funciona sem leak de recursos
- [ ] Latencia por stage e medida e acessivel
- [ ] Falha em um stage nao derruba o pipeline
- [ ] Teste: pipeline com stages mock (echo) funciona end-to-end

**Complexidade: Media**
| Eixo | Nivel |
|------|-------|
| Conceitos simultaneos | 4 (async tasks, channels, lifecycle, metrics) |
| Decisoes nao-obvias | 2 (backpressure, error recovery) |
| Fronteiras cruzadas | 2 (pipeline + audio crates) |
| Comportamento alterado | Novo codigo |

---

### T-06: Speech-to-Text — Whisper.cpp Integration

**Objetivo:** Transcrever AudioChunks em TextSegments usando Whisper.cpp local.

**Depende de:** T-01

**Escopo:**
- Integrar whisper-rs (binding para whisper.cpp)
- Carregar modelo `base.en` ao iniciar
- Receber `AudioChunk`, retornar `TextSegment`
- Implementar como `PipelineStage`
- Download automatico do modelo na primeira execucao (ou instrucoes manuais)

**Criterios de Aceite:**
- [ ] Modelo Whisper carrega em memoria (< 500MB RAM)
- [ ] Transcreve AudioChunk de 2-3s em < 1500ms (CPU)
- [ ] Retorna TextSegment com texto e idioma
- [ ] Implementa trait PipelineStage
- [ ] Teste: transcrever arquivo WAV conhecido, validar texto

**Complexidade: Media**
| Eixo | Nivel |
|------|-------|
| Conceitos simultaneos | 3 (FFI binding, model loading, inference) |
| Decisoes nao-obvias | 2 (model size tradeoff, chunk overlap) |
| Fronteiras cruzadas | 1 (stt crate) |
| Comportamento alterado | Novo codigo |

---

### T-07: Translation — CTranslate2 + Opus-MT

**Objetivo:** Traduzir TextSegments entre EN e PT usando modelos Opus-MT locais.

**Depende de:** T-01

**Escopo:**
- Integrar CTranslate2 via FFI ou CLI wrapper
- Carregar modelos Opus-MT (en->pt, pt->en)
- Receber `TextSegment`, retornar `TextSegment` traduzido
- Implementar como `PipelineStage`
- Tokenizacao com SentencePiece (dependencia do Opus-MT)

**Criterios de Aceite:**
- [ ] Modelos Opus-MT carregam (< 200MB RAM total para ambos)
- [ ] Traduz frase de ~20 palavras em < 500ms
- [ ] Qualidade aceitavel para frases coloquiais de reuniao
- [ ] Implementa trait PipelineStage
- [ ] Teste: traduzir frases conhecidas, validar output

**Complexidade: Media**
| Eixo | Nivel |
|------|-------|
| Conceitos simultaneos | 3 (CTranslate2, tokenization, model management) |
| Decisoes nao-obvias | 2 (FFI vs CLI, tokenizer setup) |
| Fronteiras cruzadas | 1 (translation crate) |
| Comportamento alterado | Novo codigo |

---

### T-08: Integration — STT + Translation + TTS no Pipeline

**Objetivo:** Conectar os estagios reais (Whisper, Opus-MT, Piper) no pipeline orchestrator.

**Depende de:** T-05, T-06, T-07, T-09

**Escopo:**
- Instanciar stages reais no pipeline orchestrator
- Configurar direcao de traducao (EN->PT para speaker)
- Tuning de buffer sizes entre stages
- Benchmark de latencia end-to-end
- Ajustes de chunk size baseado em resultados reais

**Criterios de Aceite:**
- [ ] Pipeline completo roda: audio -> texto -> traducao -> voz -> output
- [ ] Latencia end-to-end < 5 segundos
- [ ] Sem memory leaks apos 10 minutos de execucao
- [ ] Audio output inteligivel e sincronizado
- [ ] Teste: reuniao simulada de 2 minutos com audio em ingles

**Complexidade: Media**
| Eixo | Nivel |
|------|-------|
| Conceitos simultaneos | 4 (all stages, timing, tuning, debugging) |
| Decisoes nao-obvias | 3 (buffer tuning, chunk overlap, latency tradeoffs) |
| Fronteiras cruzadas | 2 (pipeline + all stage crates) |
| Comportamento alterado | Novo codigo |

---

### T-09: Text-to-Speech — Piper TTS

**Objetivo:** Sintetizar TextSegments traduzidos em AudioChunks usando Piper TTS local.

**Depende de:** T-01

**Escopo:**
- Integrar Piper TTS via FFI ou subprocess
- Carregar voz PT-BR (para traducao EN->PT)
- Receber `TextSegment`, retornar `AudioChunk` com audio sintetizado
- Sample rate de saida compativel com o output (16kHz ou 22kHz)
- Implementar como `PipelineStage`

**Criterios de Aceite:**
- [ ] Voz PT-BR carrega e sintetiza
- [ ] Sintetiza frase de ~20 palavras em < 800ms
- [ ] Audio output natural e inteligivel
- [ ] Implementa trait PipelineStage
- [ ] Teste: sintetizar frases e salvar WAV para validacao

**Complexidade: Media**
| Eixo | Nivel |
|------|-------|
| Conceitos simultaneos | 3 (Piper integration, voice config, audio format) |
| Decisoes nao-obvias | 2 (FFI vs subprocess, voice selection) |
| Fronteiras cruzadas | 1 (tts crate) |
| Comportamento alterado | Novo codigo |

---

### T-10: System Tray UI

**Objetivo:** Interface minimalista no system tray do Windows para controlar o pipeline.

**Depende de:** T-01

**Escopo:**
- System tray icon usando windows-rs ou tray-icon crate
- Menu de contexto com:
  - Toggle ativo/inativo
  - Selecao de direcao (EN->PT / PT->EN)
  - Indicador de latencia (texto)
  - Sair
- Comunicacao com pipeline via command channel
- Icone muda de cor conforme status (verde=ativo, cinza=inativo)

**Criterios de Aceite:**
- [ ] Icone aparece no system tray
- [ ] Menu abre com opcoes funcionais
- [ ] Toggle envia comando start/stop ao pipeline
- [ ] Direcao de traducao e configuravel
- [ ] Indicador de latencia atualiza a cada 5 segundos
- [ ] Sair encerra o processo limpo

**Complexidade: Baixa**
| Eixo | Nivel |
|------|-------|
| Conceitos simultaneos | 2 (Win32 tray API, command channel) |
| Decisoes nao-obvias | 1 (crate choice) |
| Fronteiras cruzadas | 1 (ui crate) |
| Comportamento alterado | Novo codigo |

---

### T-11: End-to-End MVP — Wiring + Smoke Test

**Objetivo:** Conectar UI + Pipeline + todos os stages para o fluxo completo do MVP.

**Depende de:** T-08, T-10

**Escopo:**
- `main.rs` que inicializa todos os componentes
- Carregamento de config TOML
- Carregamento de todos os modelos ao iniciar (splash/loading)
- Startup sequence: carregar modelos -> iniciar tray -> aguardar comando
- Graceful shutdown (Ctrl+C e menu Sair)
- Smoke test manual documentado

**Criterios de Aceite:**
- [ ] Aplicacao inicia, carrega modelos, mostra tray icon
- [ ] Usuario clica "Ativar" -> pipeline comeca a traduzir
- [ ] Audio de reuniao em ingles sai traduzido em portugues no VB-Cable
- [ ] "Desativar" para o pipeline limpo
- [ ] "Sair" encerra sem crash
- [ ] Latencia end-to-end < 5 segundos

**Complexidade: Baixa**
| Eixo | Nivel |
|------|-------|
| Conceitos simultaneos | 2 (wiring, lifecycle) |
| Decisoes nao-obvias | 1 (startup order) |
| Fronteiras cruzadas | 2 (all crates, mas apenas wiring) |
| Comportamento alterado | Novo codigo |

---

## Verificacao de Cobertura

| Requisito PRD | Task(s) |
|--------------|---------|
| RF-01: Captura Speaker | T-02 |
| RF-03: STT | T-06 |
| RF-04: Traducao | T-07 |
| RF-05: TTS | T-09 |
| RF-06: Audio Virtual | T-04 |
| RF-07: UI (MVP subset) | T-10 |
| RF-08: Pipeline | T-05, T-08 |
| RNF-01: Latencia | T-08 (benchmark) |
| RNF-02: Recursos | T-06, T-07, T-09 (memory targets) |
| RNF-03: Qualidade Audio | T-04 (crossfade), T-09 (TTS quality) |
| RNF-04: Privacidade | Todos (100% local by design) |
| VAD | T-03 |

**RF-02 (Microfone):** fora do MVP, Fase 2.

## Ordem de Execucao Recomendada

```
Semana 1:  T-01
Semana 2:  T-02 + T-06 + T-07 + T-09 + T-10 (paralelo — independentes)
Semana 3:  T-03 + T-04 (dependem de T-02 / T-01)
Semana 4:  T-05 (pipeline orchestrator)
Semana 5:  T-08 (integracao dos stages reais)
Semana 6:  T-11 (end-to-end + smoke test)
```
