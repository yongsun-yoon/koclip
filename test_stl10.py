import hydra
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch.nn.functional as F
from test_utils import prepare_model, get_text_embeddings

KO_TEMPLATES = [
    '{}를 찍은 사진'
]

KO_CLASSES = [
    '비행기', # airplane
    '새', # bird
    '자동차', # car
    '고양이', # cat
    '사슴', # deer
    '개', # dog
    '말', # horse
    '원숭이', # monkey
    '선박', # ship
    '트럭' # truck
]

def read_images(path_to_data):
    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        images = [Image.fromarray(i) for i in images]
    return images


def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        labels = labels.astype(int) - 1
    return labels



@hydra.main(config_path='conf', config_name='test_stl10')
def main(cfg):
    images = read_images(f'{cfg.cwd}/{cfg.images_path}')
    labels = read_labels(f'{cfg.cwd}/{cfg.labels_path}')

    clip, clip_processor = prepare_model(cfg.model.clip, cfg.model.kobert, cfg.device)
    text_embeddings = get_text_embeddings(clip, clip_processor, KO_CLASSES, KO_TEMPLATES)

    preds = []
    for i in tqdm(range(0, len(images), cfg.batch_size)):
        batch_images = images[i:i+cfg.batch_size]
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