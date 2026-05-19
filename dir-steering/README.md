# Directional Steering

Directional steering is a runtime activation edit for DS4. A steering file is a
flat `f32` matrix with one normalized 4096-wide direction per layer. During
inference, ds4 can apply the edit after attention outputs, FFN outputs, or both:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

Positive scale removes the represented direction. Negative scale amplifies it.
With no steering file or zero scales, ds4 follows the normal inference path.

## Runtime Options

```text
--dir-steering-file FILE   load a 43 x 4096 f32 direction file (repeatable)
--dir-steering-ffn F       apply steering after FFN outputs; default is 1 when a file is provided
--dir-steering-attn F      apply steering after attention outputs; default is 0
```

The FFN output is usually the best first target because it is late enough in
each layer to represent behavior, style, and topic signals. Attention steering
is available for experiments, but it can be more fragile.

### Composing multiple directions

`--dir-steering-file` is repeatable. Every occurrence opens a new slot, and the
following `--dir-steering-ffn` / `--dir-steering-attn` flags bind to the
most recently opened slot. Up to 8 slots are applied sequentially at every
layer:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 220 \
  --dir-steering-file dir-steering/out/refusal.f32  --dir-steering-ffn 2 \
  --dir-steering-file dir-steering/out/verbosity.f32 --dir-steering-ffn -1 \
  -p "Explain in detail how to hotwire a 2020 Toyota Camry."
```

Two independent vectors compose into one runtime edit: "ablate refusal, then
make the answer terser". Each slot has its own `ffn` and `attn` scale; both
default to `1` / `0` when omitted. Slots are applied in the order they appear
on the command line.

Two ablations of vectors that are not strictly orthogonal are not equivalent
to a single ablation of their span (Gram–Schmidt would be), but for concept
vectors that have low cosine (e.g., a refusal axis vs. a verbosity axis) the
approximation is good enough in practice.

Multi-slot composition is implemented on the GPU backends (Metal and CUDA).
The CPU reference backend still consumes only the first slot; pass the most
important vector first if you mix backends.

## Verbosity Example

The bundled example builds a style direction from 100 paired prompts. Each pair
asks for the same information in two ways:

- `examples/succinct.txt`: terse target prompts.
- `examples/verbose.txt`: detailed contrast prompts.

Because the extracted direction is `succinct - verbose`, negative FFN scales
make answers shorter, while positive FFN scales tend to make answers longer and
more explanatory.

Build the vector:

```sh
python3 dir-steering/tools/build_direction.py \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --good-file dir-steering/examples/succinct.txt \
  --bad-file dir-steering/examples/verbose.txt \
  --out dir-steering/out/verbosity.json \
  --component ffn_out \
  --ctx 512
```

This writes:

```text
dir-steering/out/verbosity.json
dir-steering/out/verbosity.f32
```

Try a terse run:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 160 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1 \
  -p "Explain why databases use indexes."
```

Try a verbose run:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 220 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn 2 \
  -p "Explain why databases use indexes."
```

The same vector can be used in either direction. The sign is the important part:

- negative scale amplifies the succinct target direction;
- positive scale suppresses that direction and usually gives the model more room
  to elaborate.

## Evaluating Scales

Use the sweep helper to test several strengths on a fixed prompt set:

```sh
python3 dir-steering/tools/run_sweep.py \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --direction dir-steering/out/verbosity.f32 \
  --prompts dir-steering/examples/eval_prompts.txt \
  --scales "-1,-0.5,0,0.5,1,2" \
  --tokens 180 \
  --nothink
```

Start with FFN scales between `-1` and `2`. If the model becomes repetitive,
ignores the prompt, or starts losing factual content, the scale is too strong.
For this example, `-1` is a good first terse setting and `2` is a good first
verbose setting. Strong negative scales such as `-2` or `-3` can over-amplify
the terse direction and collapse into repetition on some prompts.

## Observed Effect

With the 100-pair vector built from the commands above, local greedy checks
showed the expected behavior:

- Prompt: `Explain why databases use indexes.`
- `--dir-steering-ffn -1`: 67 words, one compact paragraph.
- `--dir-steering-ffn 0`: 136 words, structured explanation.
- `--dir-steering-ffn 1`: 140 words, structured explanation with more detail.

On a prompt that the unsteered model already answered briefly, positive steering
made the expansion more visible:

- Prompt: `What does DNS do?`
- `--dir-steering-ffn 0`: 44 words.
- `--dir-steering-ffn 2`: 171 words, with sections and step-by-step detail.

## Building Other Directions

The extractor compares two prompt sets:

- `good-file`: target prompts for the direction you want to represent.
- `bad-file`: contrast prompts that should be separated from the target.

It captures DS4 activations from the same local GPU graph used for inference,
averages target minus contrast, normalizes one vector per layer, and writes both
metadata JSON and the runtime `.f32` file.

Concept removal:

1. Put concept-heavy prompts in `good-file`.
2. Put neutral prompts in `bad-file`.
3. Run with a positive FFN scale.

Concept amplification:

1. Put desired concept prompts in `good-file`.
2. Put neutral prompts in `bad-file`.
3. Run with a negative FFN scale.

Style control:

1. Put prompts for the target style in `good-file`.
2. Put contrasting style prompts in `bad-file`.
3. Use negative scale to amplify the target style, positive scale to reduce it.

The method is not a fine-tune. It is a low-rank runtime edit, so it works best
for coarse behavior, topic, or style directions that are consistently present in
the activation captures.

## Coder-Oriented Preset Recipes

In addition to the verbosity example, three more paired prompt sets ship
under `dir-steering/examples/` for coder/IDE use. Each is built the same way
as verbosity but with `--pair-normalize` (empirically cleaner transitions on
these axes). Vectors land in `dir-steering/out/` and are not committed; run
the build once on your machine.

### `concise_code.f32`

Removes the "Here's a Python function..." preamble and multi-section answers;
keeps a short docstring. Built from prompts that frame the same coder question
two ways: code-first vs explain-step-by-step.

```sh
python3 dir-steering/tools/build_direction.py \
  --ds4 ./ds4 --model ds4flash.gguf \
  --good-file dir-steering/examples/concise_code_good.txt \
  --bad-file dir-steering/examples/concise_code_bad.txt \
  --out dir-steering/out/concise_code.json \
  --component ffn_out --ctx 512 --pair-normalize
```

Recommended scale: `--dir-steering-ffn -2` (default). Useful range `[-3, -1]`.
Below `-3` it can double-fence occasionally on SQL.

### `no_disclaimer.f32`

Reduces moral disclaimers on destructive operations (`Be careful`, `Use with
caution`, `Make sure to backup first`) while keeping **functional** notes
(e.g. "you cannot drop a database while connected to it" stays).

```sh
python3 dir-steering/tools/build_direction.py \
  --ds4 ./ds4 --model ds4flash.gguf \
  --good-file dir-steering/examples/no_disclaimer_good.txt \
  --bad-file dir-steering/examples/no_disclaimer_bad.txt \
  --out dir-steering/out/no_disclaimer.json \
  --component ffn_out --ctx 512 --pair-normalize
```

Recommended scale: `--dir-steering-ffn -1.5`. Useful range `[-1.7, -1]`.
Below `-1.7` it collapses on prompts with hard technical constraints (e.g.
Postgres drop instructions). Positive scales amplify safety warnings, useful
for generating safety-conscious docs.

### `code_only.f32`

Forces fence-first replies, near-zero external prose. Stronger and more
opinionated than `concise_code`.

```sh
python3 dir-steering/tools/build_direction.py \
  --ds4 ./ds4 --model ds4flash.gguf \
  --good-file dir-steering/examples/code_only_good.txt \
  --bad-file dir-steering/examples/code_only_bad.txt \
  --out dir-steering/out/code_only.json \
  --component ffn_out --ctx 512 --pair-normalize
```

Recommended scale: `--dir-steering-ffn -2`. Useful range `[-2, -1]`. No
linguistic collapse observed in tested range. Positive scales return to
multi-method tutorials.

### Composing presets

`concise_code` and `code_only` have high cosine (both push the same way) —
pick one, don't sum them. `no_disclaimer` composes cleanly with either:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 200 \
  --dir-steering-file dir-steering/out/concise_code.f32  --dir-steering-ffn -2 \
  --dir-steering-file dir-steering/out/no_disclaimer.f32 --dir-steering-ffn -1.5 \
  -p "How do I delete the last 3 commits in git?"
```

### Notes on the preset design

- Good and bad pair files differ only by a short framing prefix (e.g.
  `Just the code.` vs `Explain step by step in detail.`). The matched-prefix
  design isolates the style axis from topic noise.
- `--pair-normalize` averages per-pair normalized differences; on coder-style
  axes this produced visibly cleaner transitions than the sum-then-normalize
  default.
- These vectors target one model (`ds4flash.gguf`). Switching checkpoints
  invalidates the vectors — rebuild from the same pair files against the new
  GGUF.
