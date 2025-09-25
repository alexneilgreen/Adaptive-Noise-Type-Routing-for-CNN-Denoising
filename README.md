# Adaptive Noise-Type Routing for CNN Denoising

A CNN denoising system that identifies noise types and routes images through specialized denoising branches.

## Quick Start

### Test Data Loader

```bash
# Test all datasets (default)
python data_loader.py

# Test specific dataset
python data_loader.py --dataset cifar10
```

### Generate Noise Examples

```bash
# Generate 10 example images with all noise types
python noise_generator.py --generate_examples True --dataset cifar10

# Test specific noise type
python noise_generator.py --test_noise gaussian
```

### Available Options

- **Datasets**: `mnist`, `cifar10`, `cifar100`, `stl10`
- **Noise Types**: `gaussian`, `salt_pepper`, `uniform`, `poisson`, `jpeg`, `impulse`

Examples are saved to `./dataset/examples/` with both clean and noisy versions.
