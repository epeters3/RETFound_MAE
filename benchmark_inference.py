"""
A quick and dirty script that profile's the fine-tuned model performance
when performing inference on CPU-only. Run it like this: (note that most
of these arguments are not needed, but they don't hurt anything)

```
CUDA_VISIBLE_DEVICES="" python benchmark_inference.py \
    --eval \
    --batch_size 1 \
    --world_size 1 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --nb_classes 2 \
    --data_path ./IDRiD_data/
    --task ./internal_IDRiD/ \
    --resume ./finetune_IDRiD/checkpoint-best.pth \
    --num_workers 1
```
"""

from time import time

import torch
from main_finetune import get_args_parser

from util.datasets import build_dataset
import models_vit

def main():
    args = get_args_parser()
    args = args.parse_args()
    print('building dataset')
    dataset_test = build_dataset(is_train='test', args=args)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print('instantiating model')
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    print('loading checkpoint')
    checkpoint = torch.load('finetune_IDRiD/checkpoint-best.pth', map_location='cpu')
    checkpoint_model = checkpoint['model']
    print('loading state dict')
    model.load_state_dict(checkpoint_model, strict=False)
    print('model is on device:', next(model.parameters()).device)
    print(f'performing inference with batch size {args.batch_size}')
    for x, y in data_loader_test:
        print("running x through the model:", x.shape, x.dtype, x.device)
        start = time()
        y_hat = model(x)
        print(f'duration: {time() - start:.4f}s')
        print("y_hat:", y_hat.shape, y_hat.dtype, y_hat.device)

if __name__ == "__main__":
    main()
