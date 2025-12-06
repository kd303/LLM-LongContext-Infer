
# LLM-LongContext-Infer
The repository is collection of references, papers required for LongContext inferencing.

# Goals
1. Reduce TTFT lowest
2. Model size / Contet size / KV Cache strategies
3. Include Architectures - Dense, MoE challenges

## References
 - (Helix Parallelism)[https://arxiv.org/abs/2507.07120] :



 - (Medha -  Tackling Heterogeneity in Long-Context LLM Inference with Medha)[https://arxiv.org/abs/2409.17264]
  	> To tackle KV cache scaling, recent work like Medha [7] shards the KV  
	> cache across an auto-scaled      pool of N GPUs using KV Parallelism  
	> (KVP), so each device stores only a fraction of the   
	> multimillion-token history. This approach significantly reduces both  
	> per GPU cache size and read latency  during self-attention. However,  
	> Medha and similar methods then gather the attention outputs onto a    
	> fixed group of TP GPUs (e.g., 8) for all subsequent FFN computations. 
	> In effect, while KVP fans  out computation across N GPUs for   
	> attention, it does not repurpose those same GPUs to further    
	> accelerate FFN execution. As a result, FFN weight loads remain a   
	> latency bottleneck, and hardware  resources become increasingly   
	> underutilized as N grows.
