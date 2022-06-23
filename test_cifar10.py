import hydra
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch.nn.functional as F
import torchvision
from test_utils import prepare_model, get_text_embeddings


KO_TEMPLATES = [
    '이것은 {}이다.',
    '{}를 찍은 사진.',
    '{}가 담긴 이미지.'
]


KO_CLASSES = [
    '비행기', # 'airplane',
    '자동차', # 'automobile',
    '새', # bird
    '고양이', # 'cat',
    '사슴', # 'deer',
    '개', # 'dog',
    '개구리', # 'frog',
    '말', # 'horse',
    '선박', # 'ship',
    '트럭', # 'truck',
]




@hydra.main(config_path='conf', config_name='test_cifar10')
def main(cfg):
    datas = torchvision.datasets.CIFAR10(cfg.data_path, train=False, download=True)
    datas = list(datas)

    clip, clip_processor = prepare_model(cfg.model.clip, cfg.model.kobert, cfg.device)
    text_embeddings = get_text_embeddings(clip, clip_processor, KO_CLASSES, KO_TEMPLATES)

    preds, labels = [], []
    for i in tqdm(range(0, len(datas), cfg.batch_size)):
        batch = datas[i:i+cfg.batch_size]
        batch_images, batch_labels = zip(*batch)
        labels += list(batch_labels)

        inputs = clip_processor(images=list(batch_images), return_tensors="pt", padding=True).to(cfg.device)
        image_embeddings = clip.get_image_features(**inputs)
        image_embeddings = F.normalize(image_embeddings, dim=1)
        logits = image_embeddings @ text_embeddings.T
        preds += logits.argmax(dim=1).cpu().tolist()

    preds = np.array(preds)
    acc = (labels == preds).mean() * 100
    print(f'Acc: {acc:.2f}')


if __name__ == '__main__':
    main()