# Learning modern LLM structure 

Modern LLMs including : 

## Qwen-VL

- TextMonkey : Merge tokens based on token similarity
- GUI-Odyssey : Add history module (Perceiver Resampler)

## Qwen2-VL

- ShowUI : UI-Graph, tokens masked in LLM decoder.
- Adding Visualization Script of UI Graph (2025-03-06) `ShowUI/edited/image_processing_qwen2_vl.py`
  - Random UI Graph Drop
  - Uniform UI Graph Drop
- Adding UI Graph into Qwen2VL (in transformers library) support LLama-Factory Finetuning (2025-03-06)
