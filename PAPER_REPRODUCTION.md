# Paper reproduction commands

These commands demonstrate the strict model construction. Replace dataset paths
and checkpoints with the exact assets used by the corresponding paper when
reproducing reported accuracy.

## Visual prompting

```bash
python main.py \
  --tuning_method prompt \
  --backbone resnet50 \
  --weights DEFAULT \
  --dataset flowers102 \
  --data_path /path/to/data \
  --prompt_size 30 \
  --paper_hparams True
```

Only the prompt tensors are optimized. No downstream classifier is created.

## Conv-Adapter

```bash
python main.py \
  --tuning_method conv \
  --backbone resnet50 \
  --weights DEFAULT \
  --conv_adapter_mode conv_parallel \
  --dataset flowers102 \
  --data_path /path/to/data
```

## BAM

```bash
python main.py \
  --tuning_method bam \
  --backbone resnet50 \
  --weights DEFAULT \
  --dataset imagefolder \
  --data_path /path/to/imagenet \
  --paper_hparams True
```

BAM is end-to-end training, not frozen-backbone PEFT.

## Residual Adapters

```bash
python main.py \
  --tuning_method residual \
  --ra_mode parallel \
  --ra_pretrained_checkpoint /path/to/official/shared_resnet26.pth \
  --dataset imagefolder \
  --data_path /path/to/task
```

## SSF

```bash
python main.py \
  --tuning_method ssf \
  --backbone vit_b_16 \
  --weights DEFAULT \
  --dataset flowers102 \
  --data_path /path/to/data
```

## LoRA

```bash
python main.py \
  --tuning_method lora \
  --backbone vit_b_16 \
  --weights DEFAULT \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0 \
  --lora_merge_weights True \
  --dataset flowers102 \
  --data_path /path/to/data
```

## BitFit

```bash
python main.py \
  --tuning_method bitfit \
  --backbone vit_b_16 \
  --weights DEFAULT \
  --dataset flowers102 \
  --data_path /path/to/data
```

## Side-Tuning

```bash
python main.py \
  --tuning_method sidetune \
  --backbone resnet18 \
  --weights DEFAULT \
  --sidetune_alpha 0.5 \
  --dataset flowers102 \
  --data_path /path/to/data
```
