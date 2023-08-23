# [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/#kv-cache)

## kv cache

对于采样，transformer推理包括处理提供的提示/上下文(可以并行发生)，然后逐个采样额外的token(这就是自回归的表现)。在采样中，transformer执行自注意，这需要当前序列中每个token的kv值(无论是提示/上下文还是生成的token)，以称为kv缓存的矩阵的形式提供。过去的缓存形状类似 $[batch, 2, num~heads, seq~len, features]$。