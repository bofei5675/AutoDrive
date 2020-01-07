# Peking University/Baidu - Autonomous Driving
## Best Model Configuration
Current Score: **0.065**

Configuration:
```
Namespace(alpha=2, batch_size=2, beta=4, debug=False, epoch=30, gamma=10.0, loss_type='FL', model_type='HG2', num_classes=8, num_features=256, num_stacks=4, pre_train=True, prob=0.2, save_dir='run_large_val/', sigma=1, val_size=0.2)
```
Decoding with threshold=-0.5
## Some facts
Training data
1. average cars number: 11.418839558374442, sum of cars: 48610
2. Pre-trained weights are from [this repo](https://github.com/princeton-vl/pytorch_stacked_hourglass)
3. Refer to this [kaggle discussion](https://www.kaggle.com/c/pku-autonomous-driving/discussion/117621) to remove broken image to obtain `train_fixed.csv`
## Improvements
- [x] U-Nets, score: 0.028
- [x] Stacked Hourglassnetwork, score: 0.018-0.045 (based on number of stacks)
- [x] threshold for logits is tricky (previously, it's 0 but it's not always work)
- [x] Pixel-wised augmentation (implemented)
- [x] Remove broken image on training dataset (implemented) - Increase score to 0.065 for 4-stacks HG
- [ ] Normalize all image based on overall mean and standard deviation (not implemented yet)
