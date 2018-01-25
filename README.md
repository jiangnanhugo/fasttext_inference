## Fasttext
this source file is from fasttext.cc version 1.0b

## Inference
slim Inference model for mocel compression.

we remove the output_ and qoutput during inference, which require smaller footprint for device with smaller memory.
