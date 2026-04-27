# Real-Time Meeting Translator — Visão Geral da Solução

Documento-resumo para comparação com soluções equivalentes.
Foco em **arquitetura, tecnologias, performance e trade-offs**.

---

## 1. Objetivo

Tradutor simultâneo **PT ↔ EN** para reuniões online, atuando como
**proxy de áudio** no Windows: a aplicação intercepta o áudio do
Teams/Zoom/Chrome (loopback) e o áudio do microfone do usuário,
traduz nas duas direções em tempo real e devolve o áudio traduzido em
vozes neurais.

**Princípios não-negociáveis:**

| Princípio | Como se manifesta |
|---|---|
| 100% local / offline | Nenhuma chamada de API externa, nenhum custo recorrente, dados de reunião nunca saem da máquina |
| Windows-only (MVP) | WASAPI nativo + VB-Cable como mic virtual |
| Latência > Cobertura | Apenas PT↔EN, detecção de idioma manual, modelos quantizados, chunks pequenos |
| Latência alvo end-to-end | **2–5 s** (do som original ao áudio traduzido nos fones) |
| Setup zero para o ouvinte | Sem precisar reconfigurar áudio do Windows toda vez (ducking automático) |

---

## 2. Topologia de Áudio

```
┌─────────────────┐       ┌──────────────────────┐       ┌──────────────┐
│  Reunião (Zoom, │──────▶│ VB-Cable (Output     │──────▶│  Loopback    │
│  Teams, Chrome) │       │  default do Windows) │       │  Capture     │
└─────────────────┘       └──────────────────────┘       └──────┬───────┘
                                                                │
                                                  Pipeline Speaker (loopback → fones)
                                                                ▼
┌─────────────────┐       ┌──────────────────────┐       ┌──────────────┐
│ Mic físico do   │──────▶│   AudioCapture       │──────▶│ Pipeline Mic │
│ usuário         │       │   (cpal/WASAPI)      │       │ (mic → V.Mic)│
└─────────────────┘       └──────────────────────┘       └──────┬───────┘
                                                                │
                                                                ▼
                                              ┌────────────────────────────────┐
                                              │ Hi-Fi Cable (Input default     │
                                              │ do Windows) ← reunião lê daqui │
                                              └────────────────────────────────┘
```

São **dois pipelines paralelos e independentes**, ambos consumindo o mesmo
conjunto de modelos carregados em memória. A separação de **VB-Cable**
(loopback) de **Hi-Fi Cable** (mic virtual) evita que a saída TTS de uma
direção contamine a captura da outra.

---

## 3. Pipeline por Estágio

```
Capture (cpal/WASAPI) → Silero VAD → Sepformer (opt-in)
   → Streaming Whisper STT (Local Agreement-2)
   → Diarização ECAPA-TDNN + F0 (pyworld)
   → Tradução NLLB-200 via CTranslate2 int8_float16
   → Piper TTS + WORLD pitch shift
   → Mixer com ducking WASAPI → Saída
```

### 3.1 Captura — `cpal` + WASAPI loopback
- Chunks de **280 ms** (`DEFAULT_CHUNK_DURATION_MS`).
- Mono 16 kHz. Loopback nativo no Windows via `cpal::Host::WASAPI`.

### 3.2 VAD — **Silero VAD** (ONNX)
- Modelo LSTM `silero_vad.onnx`, ~1.8 MB.
- Roda em Rust nativo via `ort` (ONNX Runtime).
- Decisão por frame de 30 ms; chunk passa adiante se **qualquer** frame > 0.5.
- Substituiu um RMS energy gate que produzia falsos positivos (ar-condicionado
  acionando Whisper) e falsos negativos (fala baixa descartada).
- **Custo:** P50 ~1–2 ms, P95 ~3 ms.
- **Fallback:** se o modelo não está em disco, cai num gate de energia.

### 3.3 Source Separation — **Sepformer-libri2mix** (opt-in)
- Quando ligado, divide cada chunk em 2 canais → 2 pipelines paralelas.
- Resolve o caso clássico de **interrupções** ("salada" do Whisper quando
  duas vozes se sobrepõem).
- **Custo:** ~50–80 ms/chunk em GPU, ~120 MB de pesos.
- **Default OFF** porque em conversas 1-a-1 o custo é desperdiçado.

### 3.4 STT — **Whisper.cpp nativo (whisper-rs)**, *streaming*
- Modelo `ggml-small-q5_1.bin` (quantização **q5_1**, ~181 MB vs 466 MB fp16).
- Roda em GPU via CUDA, mas funciona em CPU.
- **Streaming via Local Agreement-2** (Macháček et al. 2023):
  - Janela rolante de 8 s, inferência a cada 250 ms.
  - O **prefixo comum** entre duas execuções consecutivas é "comprometido"
    e enviado downstream; a cauda fica em hold até confirmação.
  - Janela trim para 3 s quando ultrapassa 6 s comprometidos.
- Anti-alucinação: temperatura 0, `no_speech_thold=0.5`, `entropy_thold=2.4`.
- Idioma **forçado** por chunk (sem auto-detect) para evitar Whisper inventar
  texto na língua errada.
- `WhisperState` é alocado **uma vez** e reutilizado — economiza ~460 MB de
  realocação de buffers GPU por chunk.

### 3.5 Diarização — **SpeechBrain ECAPA-TDNN** + **pyworld F0**
- Modelo `speechbrain/spkrec-ecapa-voxceleb`, embedding 192-d.
- Open weights (sem token HF).
- Threshold cosseno 0.55 (margem confortável: intra ~0.65–0.85, inter ~0.10–0.45).
- **Confidence smoothing:** novo speaker_id só vira "ativo" após 2 chunks
  consecutivos — evita flush destrutivo no meio de frase por wobble single-chunk.
- **F0 tracking:** pyworld DIO extrai a fundamental por chunk; média rolante
  por speaker_id alimenta o pitch shift do TTS.

### 3.6 Tradução — **NLLB-200-distilled-600M** via **CTranslate2 int8_float16**
- Modelo único bidirecional via `target_prefix` (códigos FLORES-200
  `eng_Latn` / `por_Latn`).
- Beam=5, `length_penalty=1.0`, `no_repeat_ngram=3`.
- **Custo:** ~150–200 ms/sentença em GPU int8_float16.
- Substituiu Opus-MT (commit 94099c1) — output mais natural e context-aware,
  custo equivalente.
- **Não é LLM autoregressivo de propósito geral** — é encoder-decoder
  especializado em tradução. Sem alucinação criativa.

### 3.7 TTS — **Piper** + **WORLD analysis-synthesis** (pyworld)
- Piper sintetiza em voz fixa por idioma (Faber-medium pt-BR, Ryan-medium en-US).
- ONNX, **roda em CPU**, ~150 ms/utterance, ~25 MB por voz.
- pyworld extrai F0, envelope espectral e aperiodicidade do output do Piper,
  troca a F0 pela média rolante do speaker, aplica formant shift conservador
  (clamp 0.85–1.15) e ressintetiza.
- Total TTS: **~250–350 ms** com pitch shift.
- **Trade-off explícito:** a voz NÃO clona o speaker — só mantém um
  *cue* de "quem está falando agora" via pitch + formant. Para clonar de
  fato seria preciso CosyVoice 2-0.5B (testado e descartado: 2–4 s/utterance
  numa RTX 3050 6 GB, fora do orçamento de latência).

### 3.8 Mixer + Ducking — **WASAPI Session API**
- Mixer único combina passthrough do áudio original com TTS.
- **Ducking automático:** quando TTS está tocando, enumera todas as sessões
  de áudio do endpoint default e baixa volume para 25% via
  `ISimpleAudioVolume::SetMasterVolume`. Restaura ao silêncio do TTS.
- Nossa própria sessão é **excluída** do ducking.
- Thread COM dedicada (`CoInitializeEx(COINIT_MULTITHREADED)`).

---

## 4. Stack & Linguagens

| Camada | Tecnologia | Motivo |
|---|---|---|
| Núcleo / orquestração | **Rust** (Tokio async) | Performance, zero-cost abstractions, FFI direto com whisper.cpp e ONNX Runtime |
| STT runtime | whisper.cpp (via `whisper-rs`) | Único do mercado com q5_1 + GPU CUDA + qualidade PT-BR |
| ML runtime (VAD) | ONNX Runtime (`ort`) | Inferência leve em Rust nativo, sem Python no caminho do áudio |
| Tradução runtime | CTranslate2 (Python bridge) | 3–5× throughput vs HuggingFace transformers; int8 com perda chrF <1 |
| TTS runtime | Piper ONNX (Python bridge) | TTS neural <200 ms em CPU, sem GPU |
| Diarização | SpeechBrain ECAPA-TDNN (Python bridge) | Open weights, embedding com margem confortável |
| Source separation | Sepformer libri2mix (Python bridge) | Estado-da-arte para 2 falantes simultâneos |
| Áudio I/O | `cpal` + WASAPI | Loopback nativo Windows, low-level |
| Roteamento Windows | VB-Cable + Hi-Fi Cable | Mic virtual / saída virtual, sem precisar de driver custom |
| UI | System tray (Win32 nativo) | Footprint mínimo, estilo notificação |

**Observação importante:** todas as pontes Python rodam como **subprocessos
persistentes** com protocolo JSON-lines (texto) ou framed-binary (TTS).
Modelos são carregados **uma vez** no boot do bridge; cada chunk paga só o
custo de inferência. Tokio + `spawn_blocking` evita bloquear o runtime async
durante I/O com Python.

---

## 5. Performance — Números Reais (RTX 3050 6 GB, i5-11400H)

| Estágio | P50 | P95 | Cota do orçamento |
|---|---|---|---|
| VAD (Silero) | 1–2 ms | 3 ms | trivial |
| Separação (Sepformer, GPU, opt-in) | 50 ms | 80 ms | opcional |
| STT (Whisper small q5_1, streaming) | ~250 ms / inferência (a cada 250 ms) | — | dominante |
| Diarização (ECAPA + F0) | 30–50 ms | 70 ms | secundário |
| Tradução (NLLB int8_float16) | 150 ms | 250 ms | secundário |
| TTS (Piper + pyworld) | 250 ms | 400 ms | secundário |
| Ducking (debounced) | <5 ms | — | trivial |
| **End-to-end** (primeira palavra → áudio nos fones) | **~1.5–2 s** | **~3 s** | dentro do alvo de 2–5 s |

**Onde o tempo realmente gasta:** STT é o estágio dominante. O Local Agreement-2
**não reduz** o custo de Whisper — pelo contrário, roda Whisper ~3–4× mais
vezes do que um pipeline batch-mode rodaria. O ganho está em **time-to-first-word**:
com batch, a primeira palavra só sai depois de ~2.5 s acumulados; com streaming,
ela sai ~1 s depois do speaker começar a falar.

**VRAM em uso:** ~2 GB (Whisper) + ~1.2 GB (NLLB CT2) + ~120 MB (Sepformer
opt-in). Cabe folgado em GPU 6 GB. TTS roda em CPU.

---

## 6. Trade-offs Conscientes

### 6.1 Latência > Cobertura
- **Apenas PT↔EN.** Adicionar idiomas exigiria modelos extras na VRAM e
  *language ID* (mais um estágio). Decisão de produto: o usuário-alvo só
  fala português e participa de reuniões em inglês.
- **Sem auto-detecção de idioma.** Whisper recebe a língua *forçada* por
  chunk. Auto-detect adicionaria 50–100 ms e abriria espaço para alucinação
  cross-lingual (Whisper "vendo" inglês onde tem português).

### 6.2 Vozes Diferenciadas, NÃO Clonadas
- **CosyVoice 2 foi testado e abandonado** (ADR 0006). Cloning faithful custa
  2–4 s/utterance num GPU consumer — quebra o orçamento.
- A solução adotada (Piper + WORLD pitch shift) dá cue suficiente para
  *"tell who's speaking"* mas dois homens adultos com F0 ~110 Hz vão soar
  iguais. Aceito.

### 6.3 Streaming STT Custa GPU Extra
- Local Agreement-2 roda Whisper ~3–4× mais que batch-mode. Compensado por
  remover CosyVoice (que sozinho dominava ~2 s/utterance).
- Resultado líquido: latência menor com utilização GPU equivalente.

### 6.4 Sepformer É Opt-In
- Conversas 1-a-1 não pagam o custo. Reuniões com interrupções frequentes
  ligam a flag e aceitam +50–80 ms/chunk + 120 MB.

### 6.5 Quantização Int8
- Whisper q5_1: chrF cai <1 ponto vs fp16 em PT/EN.
- NLLB CT2 int8_float16: idem.
- Aceito: ganho de 2× em velocidade + 4× em disco.

### 6.6 Ducking Mexe no Mixer do Windows
- `ISimpleAudioVolume::SetMasterVolume` altera o slider de volume por
  sessão visível no Mixer do Windows. Restaurado no `Drop`, mas um crash
  pode deixar outros apps em volume reduzido até o usuário restaurar manualmente.
- Alternativa rejeitada: `AudioCategory_Communications` — Windows aplica
  ducking implícito não-tunável (80% fixo, transições abruptas).

### 6.7 Acoplamento ao Windows
- WASAPI loopback, VB-Cable, Win32 tray, `windows-sys` crate. Portar para
  macOS/Linux exigiria reescrever toda a camada de áudio + roteamento.
  Sem prioridade — público-alvo é 100% Windows.

### 6.8 Bridges Python (não-eliminado)
- Tradução, TTS, diarização e separação ainda passam por subprocessos Python.
  Trabalho futuro: portar tradução para `tokenizers-rs` + ONNX (estimado:
  semanas). STT já foi portado nativamente (whisper.cpp).

---

## 7. Decisões Arquiteturais Documentadas (ADRs)

| ADR | Decisão | Arquivo |
|---|---|---|
| 0001 | Tradução via CTranslate2 int8 | `docs/adr/0001-ctranslate2-translation.md` |
| 0002 | Audio ducking via WASAPI | `docs/adr/0002-audio-ducking-wasapi.md` |
| 0003 | CosyVoice 2 (depois superado por 0006) | `docs/adr/0003-cosyvoice-tts.md` |
| 0004 | Streaming STT com Local Agreement-2 | `docs/adr/0004-streaming-stt-local-agreement.md` |
| 0005 | Diarização ECAPA-TDNN | `docs/adr/0005-ecapa-tdnn-diarization.md` |
| 0006 | Piper + pyworld pitch shift (substitui CosyVoice) | `docs/adr/0006-piper-pitch-shift-tts.md` |
| 0007 | Sepformer source separation (opt-in) | `docs/adr/0007-sepformer-source-separation.md` |
| 0008 | Silero VAD (substitui RMS gate) | `docs/adr/0008-silero-vad.md` |

Cada ADR contém: contexto, decisão, alternativas consideradas, consequências
positivas/negativas/neutras, plano de rollout e plano de rollback.

---

## 8. Resumo em Uma Tela

- **Stack:** Rust + Python bridges, tudo offline, Windows-only.
- **STT:** Whisper small q5_1 nativo, streaming com Local Agreement-2.
- **Tradução:** NLLB-200-distilled-600M via CTranslate2 int8_float16.
- **TTS:** Piper (CPU) + pyworld pitch shift por speaker.
- **VAD:** Silero ONNX nativo (Rust).
- **Diarização:** ECAPA-TDNN + F0 com confidence smoothing.
- **Separation:** Sepformer libri2mix (opt-in).
- **Áudio:** WASAPI loopback + VB-Cable + Hi-Fi Cable, ducking automático.
- **Latência:** P50 ~1.5–2 s end-to-end, P95 ~3 s.
- **Custo:** zero (sem API, sem cloud).
- **Hardware mínimo:** GPU 4 GB VRAM (idealmente 6 GB), CPU moderno.
- **Trade-off central:** vozes neurais diferenciadas por pitch, NÃO clonadas;
  apenas PT↔EN; Windows-only.
