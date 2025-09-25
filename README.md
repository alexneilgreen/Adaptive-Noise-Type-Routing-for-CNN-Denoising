# Adaptive Noise-Type Routing for CNN Denoising

A CNN denoising system that identifies noise types and routes images through specialized denoising branches.

## Quick Start

**Important:** Run `data_loader.py` first to download datasets before using `noise_generator.py`

### 1. Download Datasets

```bash
# Download all datasets (default)
python data_loader.py

# Download specific dataset
python data_loader.py --dataset mnist
```

### 2. Generate Noise Examples

```bash
# Generate examples for all datasets (default)
python noise_generator.py --generate_examples True

# Generate examples for specific dataset
python noise_generator.py --generate_examples True --dataset cifar10
```

## File Arguments

### data_loader.py

```bash
python data_loader.py [OPTIONS]
```

**Arguments:**

- `--dataset` : Dataset to load (default: `all`)
  - Choices: `all`, `mnist`, `cifar10`, `cifar100`, `stl10`
- `--batch_size` : Batch size for testing (default: `4`)

**Examples:**

```bash
python data_loader.py --dataset all --batch_size 8
python data_loader.py --dataset mnist --batch_size 16
```

### noise_generator.py

```bash
python noise_generator.py [OPTIONS]
```

**Arguments:**

- `--generate_examples` : Generate example images (default: `False`)
  - Type: `bool` (True/False)
- `--dataset` : Dataset to use for examples (default: `all`)
  - Choices: `all`, `mnist`, `cifar10`, `cifar100`, `stl10`
- `--num_examples` : Number of example images to generate (default: `10`)
- `--output_dir` : Directory to save examples (default: `./noise_examples`)
- `--test_noise` : Test a specific noise type
  - Choices: `gaussian`, `salt_pepper`, `uniform`, `poisson`, `jpeg`, `impulse`

**Examples:**

```bash
# Generate examples for all datasets
python noise_generator.py --generate_examples True --num_examples 5

# Generate examples for MNIST only
python noise_generator.py --generate_examples True --dataset mnist --output_dir ./mnist_examples

# Test specific noise type
python noise_generator.py --test_noise gaussian

# Show available noise parameters (default behavior)
python noise_generator.py
```

## Available Options

### Datasets

- `mnist` - MNIST handwritten digits
- `cifar10` - CIFAR-10 object recognition
- `cifar100` - CIFAR-100 object recognition
- `stl10` - STL-10 object recognition
- `all` - All datasets above

### Noise Types

- `gaussian` - Gaussian/normal noise
- `salt_pepper` - Salt and pepper noise
- `uniform` - Uniform random noise
- `poisson` - Poisson noise
- `jpeg` - JPEG compression artifacts
- `impulse` - Random-valued impulse noise

## Output Structure

When generating examples with `--dataset all`, the output structure is:

```
noise_examples/
├── mnist/
│   ├── mnist_example_1/
│   │   ├── 1_original.png
│   │   ├── 1_gaussian.png
│   │   ├── 1_salt_pepper.png
│   │   ├── ... (all noise types)
│   │   └── 1_comparison_grid.png
│   └── mnist_example_2/...
├── cifar10/...
├── cifar100/...
└── stl10/...
```

Each example includes individual noise images and a comparison grid showing the original image (left column) with all 6 noise variations.
