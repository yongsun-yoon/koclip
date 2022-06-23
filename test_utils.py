import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel


def prepare_model(clip_model_name_or_path, kobert_model_name_or_path, device):
    clip = CLIPModel.from_pretrained(clip_model_name_or_path)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name_or_path)

    kobert = AutoModel.from_pretrained(kobert_model_name_or_path)
    kobert_tokenizer = AutoTokenizer.from_pretrained(kobert_model_name_or_path)

    kobert.pooler.dense.weight.data = kobert.pooler.dense.weight[:clip.projection_dim].data
    kobert.pooler.dense.bias.data = kobert.pooler.dense.bias[:clip.projection_dim].data

    clip.text_model = kobert
    clip_processor.tokenizer = kobert_tokenizer

    _ = clip.requires_grad_(False).eval().to(device)
    return clip, clip_processor


def get_text_embeddings(clip, clip_processor, classes, templates):
    text_embeddings = []
    for cls in classes:
        text = [tpl.format(cls) for tpl in templates]
        inputs = clip_processor(text=text, padding=True, return_tensors='pt').to(clip.device)
        emb = clip.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        emb = F.normalize(emb, dim=1)
        emb = emb.mean(dim=0, keepdims=True)
        emb = F.normalize(emb, dim=1)
        text_embeddings.append(emb)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    return text_embeddings
        

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]