# Peking University/Baidu - Autonomous Driving
## Some facts
Training data
1. average cars number: 11.418839558374442, sum of cars: 48610
2. Pre-trained weights are from [this repo](https://github.com/princeton-vl/pytorch_stacked_hourglass)
## Improvements
- [x] U-Nets, score: 0.028
- [x] Stacked Hourglassnetwork, score: 0.018-0.045 (based on number of stacks)
- [ ] threshold for logits is tricky (previously, it's 0 but it's not always work)
- [ ] Pixel-wised augmentation (implemented)
- [ ] Remove broken image on training dataset (implemented)
- [ ] Normalize all image based on overall mean and standard deviation (not implemented yet)
