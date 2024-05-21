import torch
import torch.nn as nn

class RQ_Prompt(nn.Module):
    def __init__(self, length, embed_dim, embedding_key, pool_size, top_k, prompt_init='uniform', prompt_pool=True,
                 prompt_key=True, batchwise_prompt=True, prompt_key_init='uniform', ):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k

        prompt_pool_shape = (pool_size, length, embed_dim)
        nn.init.uniform_(self.prompt, -1, 1)

        key_shape = (pool_size, embed_dim)
        self.prompt_key = nn.Parameter(torch.randn(key_shape))
        nn.init.uniform_(self.prompt_key, -1, 1)

        residual_prompt_pool_shape = (pool_size, length, embed_dim)
        self.residual_prompt = nn.Parameter(torch.randn(residual_prompt_pool_shape))
        nn.init.uniform_(self.residual_prompt, -1, 1)

        self.residual_prompt_key = nn.Parameter(torch.randn(residual_key_shape))
        nn.init.uniform_(self.residual_prompt_key, -1, 1)

    def forward(self, x_embed, iseval, task_id):
        out = dict()

        if self.embedding_key == 'mean':
            x_embed_mean = torch.mean(x_embed, dim=1)
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")

        prompt_norm = self.l2_normalize(self.prompt_key, dim=1)  # Pool_size, C
        
        similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool_size
        _, idx = torch.topk(similarity, k=self.top_k, dim=1)
        
        batched_prompt_raw = self.prompt[idx]  # B, top_k, length, C

        batch_size, top_k, length, c = batched_prompt_raw.shape

        # Put pull_constraint loss calculation inside

        # Reidual Prompt
        resdiual_prompt_key_norm = self.l2_normalize(self.residual_prompt_key, dim=1)  # Pool_size, C
        resdiual = self.prompt_key[idx].squeeze(1) -  x_embed_norm.squeeze(1)  # B, C
        resdiual_similarity = torch.matmul(resdiual, resdiual_prompt_key_norm.t())  # B, Pool_size

        resdiual_batched_prompt_raw = self.residual_prompt[resdiual_idx]  # B, top_k, length, C

        batch_size, top_k, length, c = resdiual_batched_prompt_raw.shape
        resdiual_batched_prompt = resdiual_batched_prompt_raw.reshape(batch_size, top_k * length, c)  # B, top_k * length, C

        # batched_prompt = batched_prompt.detach()
        out['prompted_embedding'] = torch.cat([resdiual_batched_prompt, batched_prompt, x_embed], dim=1)

        # Put pull_constraint loss calculation inside
        resdiual_reduce_sim =  self.reduce_similarity(resdiual, resdiual_prompt_key_norm, resdiual_idx)


        return out['prompted_embedding'], out['reduce_sim']

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        return x * x_inv_norm
    
    def reduce_similarity(self, x_embed_norm, prompt_key_norm, idx):
        batched_key_norm = prompt_key_norm[idx]  # B, top_k, C
        x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
        sim = batched_key_norm * x_embed_norm  # B, top_k, C
        reduce_sim = torch.sum(sim) / x_embed_norm.shape[0]  # Scalar
        return reduce_sim
