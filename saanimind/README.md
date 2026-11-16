# saanimind

Your own language model, built from scratch, that plugs into [Saani](../README.md) as a provider.

Everything here is written from first principles in PyTorch: a byte-level BPE tokenizer, a
GPT-style decoder transformer (RMSNorm, rotary positions, SwiGLU), a training loop, a sampler,
and a small server that speaks the OpenAI chat API with tool calls, so Saani uses it exactly
like Claude or OpenAI.

## What to expect

Read this before training anything.

- A model you can train on a PC is **millions** of parameters. Claude is many thousands of times
  larger. Your model will not plan and run multi-step tasks the way Claude does on day one.
- What it *can* do is learn **your** traffic: the exact prompts Saani sends, the tools it offers,
  and how the main model answers them. That is the data Saani records when `llm.record_traces`
  is on. Train on it, and the model gets better at precisely the tasks you give Saani.
- The sensible path is to let Saani keep Claude for planning and give your model the cheap
  **`fast`** tier first (short jobs, summaries, routine tool picks). Measure, then widen.
- The built-in **demo set** proves the pipeline, not the model. A nano model (0.4M parameters)
  reaches 100% correct first tool calls on held-out demo tasks after 4 minutes on a 2-core CPU,
  but asked a real Saani task it answered "Blue". The code works; it needs your data.

## Install

```powershell
cd saanimind
python -m venv .venv
.venv\Scripts\python -m pip install -e .          # installs PyTorch (CPU build by default)
```

For an NVIDIA GPU install the CUDA build of PyTorch first (see pytorch.org), then run the line above.

## Train your first model (demo data, minutes on a CPU)

```powershell
.venv\Scripts\saanimind prepare --demo --out data            # tokenizer + packed dataset
.venv\Scripts\saanimind train --data data --out runs\nano --size nano --minutes 5
.venv\Scripts\saanimind eval runs\nano                       # tool choice on held-out demo tasks
.venv\Scripts\saanimind chat runs\nano                       # talk to it
```

## Train on your real Saani data

1. In Saani, turn on trace recording: `saani config set llm.record_traces true` (or answer "y" in
   `saani init`). Every model exchange, with secrets scrubbed, is appended under `~/.saani/traces/`.
2. Use Saani normally for a while. More varied tasks make better data.
3. Build the dataset from the traces (and any notes or docs of yours):

   ```powershell
   .venv\Scripts\saanimind prepare --traces %USERPROFILE%\.saani\traces --text D:\notes --out data
   ```
4. Train. Sizes: `nano` (0.4M), `tiny` (6M), `small` (30M), `base` (110M). On a CPU stay at tiny;
   small and base want a GPU with 8 GB or more.

   ```powershell
   .venv\Scripts\saanimind train --data data --out runs\tiny --size tiny --steps 5000
   ```
   `--minutes N` stops on time, `--resume` continues a run, `--device cuda` forces the GPU.

## Attach it to Saani

Start the server (keep the window open, or add it to start-at-login yourself):

```powershell
.venv\Scripts\saanimind serve runs\tiny --port 8900
```

Then either:

- **Hybrid (recommended):** Claude keeps planning, your model takes the fast tier.
  `saani config set llm.own.tiers '["fast"]'` and run `saani doctor`: it shows the "Own model" line.
- **Everything:** `saani init`, choose option 5 (your own model). Expect weak results until the
  model has trained on a lot of your traces.

Saani talks to `http://127.0.0.1:8900/v1`. Change it with `saani config set llm.own.base_url ...`.

## How the pieces fit

| File | What it is |
|---|---|
| `tokenizer.py` | Byte-level BPE. 256 byte tokens + learned merges + 8 special tokens for conversation structure. |
| `model.py` | The transformer. `PRESETS` holds the four sizes. |
| `data.py` | Conversation format, loss masking (learn only the assistant's tokens), sources: Saani traces, text files, demo set. |
| `train.py` | AdamW, warm-up + cosine schedule, grad clipping, mixed precision on CUDA, checkpoint on best validation loss, resume. |
| `generate.py` | Load a run, build a prompt, sample with top-p, parse `<\|call\|>` JSON into tool calls. |
| `server.py` | `/v1/chat/completions`, `/v1/models`, `/health`. Standard library only, local only, one generation at a time. |
| `cli.py` | `prepare`, `train`, `info`, `chat`, `serve`, `eval`. |

Conversation format the model learns:

```
<|sys|>system prompt + tool list<|usr|>task<|ast|><|call|>{"name":"read_file","args":{"path":"a.txt"}}<|end|>
<|tool|>file contents<|ast|>answer text<|end|><|eos|>
```

## Tests

`python -m pytest tests` runs 20 tests: tokenizer round trips, causal masking, loss masking,
trace conversion, a real train-serve-request loop on a model trained for 60 steps, and bad input
handling in the server. They take about 5 seconds on a CPU.

## Limits

- No streaming, one request at a time, no API key: this server is for `127.0.0.1` only. Do not
  bind it to a public address.
- The context length is the preset's `block_size` (256 to 2048 tokens). Saani's full prompts are
  longer than a nano model's context; `tiny` and above fit more. Overflow keeps the end of the
  prompt, which is where the task is.
- Mixed precision runs on CUDA only. Apple MPS and CPU train in full precision.
- Trained and tested on Linux CPU. CUDA training and Windows are untested in this build.
