# TheBrain v13

TheBrain v13 is an experimental local conversational AI and malware-analysis
application written for Windows 2000 and later.

The project is implemented in ANSI C (C89) with x86 inline assembly and
runtime-selected SSE/SSE2 optimizations. It includes a small transformer,
BPE tokenizer, conversational training pipeline, Win32 GUI, PE inspection,
entropy analysis, import analysis, and lightweight x86 disassembly.

![TheBrain v13 interface](docs/thebrain-v13-screenshot.png)

## Important project status

This is unfinished experimental software.

- Generated replies can still be confused, repetitive, incomplete, or wrong.
- The application may occasionally crash, especially during large training
  runs or when memory is constrained.
- The model is small and is not comparable to a modern cloud language model.
- Malware-analysis results are heuristic and must not be treated as a final
  security verdict.

The repository is being published now so the source code and training work are
not lost. Bug reports and improvements are welcome.

## Main features

- C89 codebase targeting Windows 2000+
- Win32 native GUI
- Transformer language model implemented from scratch
- BPE tokenizer with English, Arabic, and French training data
- Autoregressive conversational generation
- AdamW training and checkpoint support
- Runtime CPU detection
- SSE/SSE2 accelerated operations with scalar and x87 assembly fallbacks
- Multi-core Win32 worker pool
- PE header, section, import, entropy, and suspicious API analysis
- Training corpus generator in `tools/gen_conv_corpus.py`

## Repository contents

- `brain_partA.c`, `brain_partB.c`, `brain.h` — application and Win32 GUI
- `model.c`, `model.h` — transformer model and checkpoint format
- `tokenizer.c`, `tokenizer.h` — multilingual BPE tokenizer
- `train.c`, `train.h` — model training
- `converse.c`, `converse.h` — conversational inference
- `ops.c`, `ops_mt.c` — math operators, SIMD/assembly, and multithreading
- `tensor.c`, `graph.c` — tensor and computation graph support
- `data/conv/` — conversational training data
- `data/safe/`, `data/dangerous/` — code samples used by analysis/training
- `data/text/` — security and Windows reference text
- `tools/` — corpus generation tools

Generated logs, executables, object files, tokenizers, embeddings, checkpoints,
and model binaries are intentionally excluded from Git.

## Build

Install a Visual C++ toolchain that can target 32-bit Windows, then run:

```bat
build.bat
```

Alternatively, open a configured Visual Studio developer command prompt and
run:

```bat
nmake
```

The build uses the Win32 API and targets `_WIN32_WINNT=0x0500`.

## Training

Start the program and run:

```text
easytrain data 50
```

For a longer run:

```text
easytrain data 150
```

Training creates local files such as `vocab.bpak` and `model_v13.bin`. These
files are machine-generated and are not committed.

To regenerate the additional conversational corpus:

```bat
python tools\gen_conv_corpus.py
```

More epochs cannot compensate for poor or insufficient data. Improving the
quality and variety of `.conv` examples is generally more useful than simply
running the same small dataset for longer.

## Conversation data format

Training conversations use alternating user and assistant records:

```text
U: Write a C89 hello world program.
A: #include <stdio.h> int main(void) { puts("Hello"); return 0; }
```

UTF-8 is used for multilingual files.

## Compatibility

The code is designed around C89 and Windows 2000-era APIs. Some development
toolchains may emit compatibility or deprecated-CRT warnings. Modern compiler
features should be avoided unless an equivalent C89/Windows 2000 fallback is
provided.

## Security notice

The `data/dangerous` directory contains source examples and suspicious API
patterns for defensive malware-analysis training. Treat unknown binaries and
samples as untrusted, and perform dynamic analysis only in an isolated
environment.
