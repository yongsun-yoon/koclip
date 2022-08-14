# koclip


## Evaluation
Zero-shot classification
|         vision model         |          text model          | CIFAR10 | CIFAR100 | STL10 |
|:----------------------------:|:----------------------------:|:-------:|:--------:|:-----:|
| openai/clip-vit-base-patch32 | openai/clip-vit-base-patch32 |   91.3  |   65.1   |  97.2 |
| openai/clip-vit-base-patch32 |       klue/roberta-base      |   86.6  |   33.6   |  94.9 |
| openai/clip-vit-base-patch32 | respect5716/koenbert-base    |   83.9  |   33.6   |  94.9 |


## Reference
* https://github.com/openai/CLIP
* https://huggingface.co/docs/transformers/model_doc/clip
* https://github.com/FreddeFrallan/Multilingual-CLIP
* https://arxiv.org/abs/2004.09813