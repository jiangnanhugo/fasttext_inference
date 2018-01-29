## Fasttext
this source file is from fasttext.cc version 1.0b

## Inference
slim Inference model for mocel compression.

we remove the output_ and qoutput_ during inference, which require smaller footprint for device with smaller memory.


# 1. generated slimed dumpfile
``bash
./change.sh train
fasttext  -input xxx -output xxx -slim
``
generated slimed dumpfile and remove `output_` and `qoutput_`.



``bash
./change.sh inference
``
it will generate `fasttext_inference`, and only `print-word-vectors` function can be called.


