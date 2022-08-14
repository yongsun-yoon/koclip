import math
import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from pytorch_lightning.lite import LightningLite
from transformers import AutoTokenizer, AutoModel, CLIPModel


def prepare_data(cfg):
    train_data = []
    for d in cfg.train_data_files:
        train_data.append(pd.read_csv(f'{cfg.cwd}/{d}'))
    train_data = pd.concat(train_data, ignore_index=True)
    return train_data


def get_param_groups(model, weight_decay):
    no_decay = ["bias", "bn", "ln", "norm"]
    param_groups = [
        {
            # apply weight decay
            "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": weight_decay
        },
        {
            # not apply weight decay
            "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return param_groups


def prepare_model(cfg):
    clip = CLIPModel.from_pretrained(cfg.model.clip).requires_grad_(False).eval()
    clip_tokenizer = AutoTokenizer.from_pretrained(cfg.model.clip)
    kobert = AutoModel.from_pretrained(cfg.model.kobert)
    kobert_tokenizer = AutoTokenizer.from_pretrained(cfg.model.kobert)

    params = get_param_groups(kobert, cfg.optim.weight_decay)
    optimizer = torch.optim.AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    return clip, clip_tokenizer, kobert, kobert_tokenizer, optimizer


def kl_div_loss(s, t, temperature):
    if len(s.size()) != 2:
        s = s.view(-1, s.size(-1))
        t = t.view(-1, t.size(-1))

    s = F.log_softmax(s / temperature, dim=-1)
    t = F.softmax(t / temperature, dim=-1)
    return F.kl_div(s, t, reduction='batchmean') * (temperature ** 2)


def transpose_for_scores(h, num_heads):
    batch_size, seq_length, dim = h.size()
    head_size = dim // num_heads
    h = h.view(batch_size, seq_length, num_heads, head_size)
    return h.permute(0, 2, 1, 3) # (batch, num_heads, seq_length, head_size)


def attention(h1, h2, num_heads, attention_mask=None):
    # assert h1.size() == h2.size()
    head_size = h1.size(-1) // num_heads
    h1 = transpose_for_scores(h1, num_heads) # (batch, num_heads, seq_length, head_size)
    h2 = transpose_for_scores(h2, num_heads) # (batch, num_heads, seq_length, head_size)

    attn = torch.matmul(h1, h2.transpose(-1, -2)) # (batch_size, num_heads, seq_length, seq_length)
    attn = attn / math.sqrt(head_size)
    if attention_mask is not None:
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1 - attention_mask) * -10000.0
        attn = attn + attention_mask

    return attn


def minilm_loss_fn(s, t, num_heads=16, attention_mask=None, temperature=1.0):
    attn_t = attention(t, t, num_heads, attention_mask)
    attn_s = attention(s, s, num_heads, attention_mask)
    loss = kl_div_loss(attn_s, attn_t, temperature=temperature)
    return loss


class Lite(LightningLite):
    def all_gather_and_view(self, data, sync_grads=False):
        data = self.all_gather(data, sync_grads=sync_grads)
        data = data.view(-1, *data.size()[2:])
        return data
        
    def run(self, cfg):
        if self.is_global_zero:
            print(OmegaConf.to_yaml(cfg))

        train_data = prepare_data(cfg)
        clip, clip_tokenizer, kobert, kobert_tokenizer, optimizer = prepare_model(cfg)
        _, optimizer = self.setup(kobert, optimizer)
        _ = clip.to(self.device)

        pbar = tqdm(range(1, cfg.num_training_steps+1), disable=not self.is_global_zero)
        for st in pbar:
            batch_idxs = np.random.randint(0, len(train_data), cfg.batch_size)
            batch = train_data.iloc[batch_idxs]
            en_inputs = clip_tokenizer(batch['en'].tolist(), return_tensors='pt', padding=True).to(self.device)
            ko_inputs = kobert_tokenizer(batch['ko'].tolist(), return_tensors='pt', padding=True).to(self.device)
            koen_inputs = kobert_tokenizer(batch['en'].tolist(), return_tensors='pt', padding=True).to(self.device)

            en_embeds = clip.text_model(**en_inputs).pooler_output
            ko_embeds = kobert(**ko_inputs).pooler_output[:, :clip.projection_dim]
            koen_embeds = kobert(**koen_inputs).pooler_output[:, :clip.projection_dim]            

            mse_loss = F.mse_loss(ko_embeds, en_embeds) + F.mse_loss(koen_embeds, en_embeds)
            cossim_labels = torch.ones(len(batch_idxs)).to(self.device)
            cossim_loss = F.cosine_embedding_loss(ko_embeds, en_embeds, cossim_labels) + F.cosine_embedding_loss(koen_embeds, en_embeds, cossim_labels)
            loss = cossim_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.is_global_zero:
                pbar.set_postfix({'loss': loss.item(), 'mse': mse_loss.item(), 'cossim': cossim_loss.item()})
                if st % 1000 == 0:
                    kobert.save_pretrained(cfg.ckpt_path)
                    kobert_tokenizer.save_pretrained(cfg.ckpt_path)



@hydra.main(config_path='conf', config_name='train')
def main(cfg):
    Lite(**cfg.lite).run(cfg)
    


if __name__ == '__main__':
    main()