import hydra
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch.nn.functional as F
import torchvision

from test_utils import prepare_model, get_text_embeddings


KO_TEMPLATES = [
    '이것은 {}이다.',
    '이것은 {}다.',
    '{}를 찍은 사진.',
    '{}을 찍은 사진.',
    '{}가 담긴 이미지.',
    '{}이 담긴 이미지.'
]


KO_CLASSES = [
    '사과', # apple
    '관상용 물고기', # aquarium_fish
    '아기', # baby
    '곰', # bear
    '비버', # beaver
    '침대', # bed
    '벌', # bee
    '딱정벌레', # beetle
    '자전거', # bicycle
    '병', # bottle
    '그릇', # bowl
    '남자 아이', # boy
    '다리', # bridge
    '버스', # bus
    '나비', # butterfly
    '낙타', # camel
    '깡통', # can
    '성', # castle
    '애벌레', # caterpillar
    '소', # cattle
    '의자', # chair
    '침팬지', # chimpanzee
    '시계', # clock
    '구름', # cloud
    '바퀴벌레', # cockroach
    '소파', # couch
    '게', # crab
    '악어', # crocodile
    '컵', # cup
    '공룡', # dinosaur
    '돌고래', # dolphin
    '코끼리', # elephant
    '넙치', # flatfish
    '숲', # forest
    '여우', # fox
    '여자 아이', # girl
    '햄스터', # hamster
    '집', # house
    '캥거루', # kangaroo
    '키보드', # keyboard
    '램프', # lamp
    '잔디 깎는 기계', # lawn mower
    '표범', # leopard
    '사자', # lion
    '도마뱀', # lizard
    '바닷가재', # lobster
    '남자', # man
    '단풍나무', # maple tree
    '오토바이', # motocycle
    '산', # mountain
    '쥐', # mouse
    '버섯', # mushroom
    '오크 나무', # oak tree
    '오렌지', # orange
    '난초', # orchid
    '수달', # otter
    '야자나무', # palm tree
    '배', # pear
    '트럭', # pickup truck
    '소나무', # pine tree
    '들판', # plane
    '접시', # plate
    '강아지', # poppy
    '호저', # porcupine
    '주머니쥐', # possum
    '토끼', # rabbit
    '너구리', # raccoon
    '가오리', # ray
    '길', # road
    '로켓', # rocket
    '장미', # rose
    '바다', # sea
    '물개', # seal
    '상어', # shark
    '땃쥐', # shrew
    '스컹크', # skunk
    '고층 건물', # skyscraper
    '달팽이', # snail
    '뱀', # snake
    '거미', # spider
    '다람쥐', # squirrel
    '전차', # streetcar
    '해바라기', # sunflower
    '피망', # sweet pepper
    '탁자', # table
    '탱크', # tank
    '전화기', # telephone
    '텔레비전', # television
    '호랑이', # tiger
    '트랙터', # tractor
    '기차', # train
    '송어', # trout
    '튤립', # tulip
    '거북이', # turtle
    '옷장', # wardrobe
    '고래', # whale
    '버드나무', # willow tree
    '늑대', # wolf
    '여자', # woman
    '지렁이', # worm
]


EN_CLASSES = [
    'apple',
    'aquarium fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'keyboard',
    'lamp',
    'lawn mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak tree',
    'orange',
    'orchid',
    'otter',
    'palm tree',
    'pear',
    'pickup truck',
    'pine tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow tree',
    'wolf',
    'woman',
    'worm',
]


EN_TEMPLATES = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]


@hydra.main(config_path='conf', config_name='test_cifar100')
def main(cfg):
    datas = torchvision.datasets.CIFAR100(cfg.data_path, train=False, download=True)
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