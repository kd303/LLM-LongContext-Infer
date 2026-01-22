# LLM-LongContext-Infer

The repository is collection of references, papers required for LongContext inferencing.

# Goals

1.  Reduce TTFT lowest
2.  Model size / Content size / KV Cache strategies
3.  Include Architectures - Dense, MoE challenges

## vLLM

### [Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)

- For long context prefill, we need to control the TTFT (time to first token) by amortizing the
  computation time of the prefill across query tokens.
- For long context decode, we need more space for KV cache to increase the batchsize (and hence
  the throughput).

### Prefill Context Parallel (under development approaches)

- **Partial Query and full key/value**: basically chunk the requests based on number of GPUs,
  calculate KV for each chunk and then collect KV across all GPUs to calculate attention for a
  given chunk or given query in that particular chunk for a given GPU, here the assumption is that
  KV can be held in the GPU memory.
- **Partial Query and partial key/value**: Here the for every chunk across GPU, each chunk partial
  K, V are calculated and then passed along using techniques like ring-attention to send/recv
  key/value vectors chunk by chunk.
- For long context decode, we need more space for KV cache to increase the batchsize (and hence
  the throughput).

### Decode Context Parallel

The core of Decoding stage is each GPU needs to calculate query wrt a large amount of Key/Value
tokens stored in KV cache across GPU. The core of decode is in CP scenario is how to shard the KV
cache across GPU.

For a model with `H` kv-heads, a request with `T` tokens in the context needs to store `H * T`
key/value tensors in the KV cache:

> 1.  If one GPU cannot hold them all, or we want to hold more requests in the KV cache, **we can
>     first shard the KV cache along the `H` dimension**, that's the plain tensor parallel
>     sharding. It's as simple as adding `-tp <num_gpus>` to the command line.
> 2.  `H` is limited (determined by the model architecture), when we continue to increase the tensor
>     parallel size, the KV cache for each GPU will be duplicated for `tp_size / H` times (so the
>     best settings here would be for an 8 attention heads, 8 GPU would not duplicate the KV cache,
>     hence more requests can fit into a single GPU). Of course, duplication is not good for
>     efficiency. Then we need to add decode context parallel to further shard the KV cache along
>     the `T` dimension. This is as simple as adding `-dcp <size>` to the command line. Note that
>     `size` does not increase the number of GPUs we need to launch, but just reduces the KV cache
>     duplication. The dcp size should lie in the range of `[1, tp_size/H]`. With larger dcp size,
>     the KV cache duplication is reduced, but the communication overhead increases.

**`dcp <size>` Examples:**

- _Here important thing to note, for example model with attention head 8, TP size 16 GPU, will
  make more sense, else DCP does not make sense... **Duh**!_
- _For DeepSeek-R1, we have 1 kv-head when MLA is enabled. The typical single-node deployment
  with `-tp 8` **causes 8x KV cache duplication**. We can consider adding `-dcp 8` to reduce the
  KV cache duplication._
- Kimi-K2, the architecture is similar to DeepSeek-R1, but with more parameters. When we deploy it
  with `-tp 16`, the KV cache duplication is 16x. We can add `-dcp 16` to _completely remove the
  KV cache duplication, at the cost of more communication overhead_. We can also add `-dcp 8` to
  reduce the KV cache duplication to 2x. _The communication overhead is smaller since the DCP
  communication only happens inside one node._

**In short, for decode context parallel, try to increase `-tp` size until you get satisfactory
performance, and then add `-dcp` to reduce the KV cache duplication.**

## References

- [Helix Parallelism](https://arxiv.org/abs/2507.07120):

<img width="948" height="345" alt="Helix" src="https://github.com/user-attachments/assets/3ef7d20b-41ef-42d9-969b-768db1cecc79" />

> Helix introduces a temporal pipeline within each layer, allowing the same set of GPUs to be
> reused across attention and FFN computation, while applying different parallelism strategies for
> each.

> Helix configures all available GPUs into a pool of N = KVP × TPA (TPA ≤ K), then shards the KV
> cache along the sequence dimension across the KVP GPUs, eliminating full-cache replication and
> cutting DRAM footprint and bandwidth demands. To avoid an expensive pre-attention All-Gather of
> queries across the KVP GPUs, Helix has each KVP GPU independently compute the full QKV
> projections.

- [Medha - Tackling Heterogeneity in Long-Context LLM Inference with Medha](https://arxiv.org/abs/2409.17264)
  - this is a great paper for heterogeneous requests (long and short serving) - a real production problem.

> To tackle KV cache scaling, recent work like Medha [7] shards the KV cache across an auto-scaled
> pool of N GPUs using KV Parallelism (KVP), so each device stores only a fraction of the
> multimillion-token history. This approach significantly reduces both per GPU cache size and read
> latency during self-attention. However, Medha and similar methods then gather the attention
> outputs onto a fixed group of TP GPUs (e.g., 8) for all subsequent FFN computations. In effect,
> while KVP fans out computation across N GPUs for attention, it does not repurpose those same GPUs
> to further accelerate FFN execution. As a result, FFN weight loads remain a latency bottleneck,
> and hardware resources become increasingly underutilized as N grows.
