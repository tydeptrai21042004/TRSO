# Dataset and protocol support


## Universal controlled runner

All entries below can be used with `python -m tools.run_fair_suite`. The runner
resolves the dataset task, records split sizes in `dataset_protocol.json`, and
creates an explicit method/backbone/task compatibility report. For `csv`, pass
`--task` explicitly. See `UNIVERSAL_FAIR_FRAMEWORK.md`.

The active router exposes **37 canonical dataset entries** and three task
types. Run:

```bash
python main.py --list_backbones
```

The command prints both available backbones and active dataset names.

## Single-label classification

### Torchvision datasets with official or repository-defined disjoint splits

| Dataset key | Classes | Split policy |
|---|---:|---|
| `cifar10` | 10 | Stratified train/validation split; official test |
| `cifar100` | 100 | Stratified train/validation split; official test |
| `mnist` | 10 | Stratified train/validation split; official test |
| `fashion_mnist` | 10 | Stratified train/validation split; official test |
| `emnist` | 62 | ByClass train/validation split; official test |
| `kmnist` | 10 | Stratified train/validation split; official test |
| `qmnist` | 10 | Train/validation split; official test |
| `usps` | 10 | Train/validation split; official test |
| `svhn` | 10 | Train/validation split; official test |
| `stl10` | 10 | Train/validation split; official test |
| `food101` | 101 | Train/validation split; official test |
| `oxfordiiitpet` | 37 | Trainval split into train/validation; official test |
| `flowers102` | 102 | Official train/validation/test |
| `stanford_cars` | 196 | Train/validation split; official test |
| `dtd` | 47 | Official train/validation/test, partition 1 |
| `fgvc_aircraft` | 100 | Trainval split into train/validation; official test |
| `gtsrb` | 43 | Train/validation split; official test |
| `pcam` | 2 | Official train/validation/test |
| `country211` | 211 | Official train/validation/test |
| `rendered_sst2` | 2 | Official train/validation/test |
| `places365` | 365 | Train-standard is split into train/validation; official validation is reserved as final test |
| `inaturalist` | Dataset metadata | 2021 train is split into train/validation; 2021 valid is reserved as final test |
| `fer2013` | 7 | Prepared train/validation/test folders; torchvision does not download it automatically |

### Datasets without an official final split in torchvision

`caltech101`, `eurosat`, and `sun397` use deterministic, disjoint 80/10/10
train/validation/test partitions. The previous validation-as-test behavior has
been removed.

### Fine-grained and list-based datasets

| Dataset key | Expected source |
|---|---|
| `cub200` | CUB-200-2011 metadata files and images |
| `nabirds` | NABirds metadata files and images |
| `stanford_dogs` | Stanford Dogs images and MATLAB split lists |
| `vtab` | Official-style `train800.txt`, `val200.txt`, and `test.txt` list files |
| `fewshot` | `annotations/train_meta.list.num_shot_<shot>.seed_<seed>`, `val_meta.list`, and `test_meta.list` |

## Generic datasets

### ImageFolder

Use `--dataset imagefolder` for ImageNet-style datasets and aliases including
`imagenet`, `imagenet_a`, `imagenet_r`, `imagenet_sketch`, and
`tiny_imagenet`.

Supported layouts include:

```text
root/
  train/class_x/*.jpg
  val/class_x/*.jpg
  test/class_x/*.jpg
```

or one class-folder tree, in which case the repository creates deterministic,
disjoint train/validation/test subsets.

### CSV

Use `--dataset csv` for all three tasks. Each split can be provided as a
separate CSV file. The image-path column and target columns are configurable
through the CLI.

Conceptual formats:

```text
# single label
image,label
images/a.jpg,3

# multi-label (one separated label field)
image,label
images/a.jpg,cat;tree;road

# regression (one separated numeric field)
image,label
images/a.jpg,0.15;0.42;0.81;0.23
```

## Multi-label datasets

| Dataset key | Targets |
|---|---|
| `coco` | Multi-hot object-category vector for each image |
| `voc2007` | Pascal VOC class-presence vector |
| `celeba` with attributes mode | 40 binary facial attributes |
| `csv` | Configurable binary columns |
| `fake` | Deterministic synthetic targets for smoke tests |

COCO can still expose the old majority-object single-label experiment only
through an explicit compatibility option. It is not described as detection or
segmentation.

## Regression datasets

| Dataset key | Targets |
|---|---|
| `celeba` with landmarks mode | Ten normalized landmark coordinates |
| `csv` | Configurable floating-point target columns |
| `fake` | Deterministic synthetic targets for smoke tests |

## Protocol safeguards

- Validation and final-test data are disjoint by default.
- A failed test split does not silently fall back to validation.
- `--allow_val_as_test` is an explicit compatibility escape hatch and should
  not be used for publication tables.
- Generated splits use a fixed seed and stratification when class labels are
  available.
- Download behavior respects `--download`.
- Final test evaluation occurs once after restoring the best validation
  checkpoint.

## Explicitly unsupported dormant protocol

The previous Visual Decathlon module was incomplete and has been replaced by a
clear `NotImplementedError`. The repository does not claim official Visual
Decathlon support until its annotation and scoring protocol are implemented.
