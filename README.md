# GeoSegamba Training

## Dataset Layout

```text
data_root/
  train/
    images/
    masks/
    geo_prior/        # optional
  val/
    images/
    masks/
    geo_prior/        # optional

LoveDA-style nested folders are also supported:

data_root/
  Train/
    Urban/
      images_png/
      masks_png/
    Rural/
      images_png/
      masks_png/
  Val/
    Urban/
      images_png/
      masks_png/
    Rural/
      images_png/
      masks_png/
```

`masks` should store class indices as single-channel label maps. For binary segmentation, use `0` and `1`. For multi-class segmentation, use `0 ... num_classes-1`.

## Train

```bash
python3 train.py \
  --data-root /path/to/dataset \
  --num-classes 6 \
  --in-channels 3 \
  --batch-size 4 \
  --val-batch-size 4 \
  --epochs 100 \
  --image-size 512 512
```

If you have a geographic prior map:

```bash
python3 train.py \
  --data-root /path/to/dataset \
  --num-classes 6 \
  --in-channels 3 \
  --use-geo-prior \
  --geo-prior-channels 1
```

For multispectral input, you can pass channel-wise normalization values:

```bash
--in-channels 4 --mean 0.485 0.456 0.406 0.5 --std 0.229 0.224 0.225 0.25
```

## Loss

Default loss is:

```text
1.0 * CrossEntropy + 0.5 * Dice
```

Optional focal term:

```bash
--focal-weight 0.25 --focal-gamma 2.0
```

## LoveDA Minimal Validation

If you want to validate the code path with the first 10 LoveDA samples, run:

```bash
python3 train.py \
  --data-root /path/to/LoveDA \
  --train-split Train \
  --val-split Val \
  --image-dirname images_png \
  --mask-dirname masks_png \
  --num-classes 7 \
  --batch-size 2 \
  --val-batch-size 2 \
  --epochs 1 \
  --image-size 512 512 \
  --max-train-samples 10 \
  --max-val-samples 10
```

This is only for quick code validation, not a meaningful experiment.
