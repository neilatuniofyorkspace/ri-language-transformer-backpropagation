# 🔬transformer-backpropagation

A visualization tool for understanding backpropagation in transformer neural networks.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

GradScope provides an interactive visualization of how gradients flow through a transformer model during backpropagation. Perfect for:
- Understanding deep learning fundamentals
- Teaching neural network concepts
- Debugging gradient flow issues
- Exploring attention mechanisms

## Features

- **Real Transformer Implementation**: Complete mini-transformer with multi-head attention
- **Live Gradient Visualization**: See gradients flow in real-time
- **Interactive Controls**: Toggle backprop, step through training
- **Training Metrics**: Loss curves, gradient norms, accuracy tracking
- **Attention Heatmaps**: Visualize what the model is "looking at"

## Installation

```bash
# Clone the repository
git clone https://github.com/..
cd ri-language-transformer-backpropagation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

```bash
# Run the visualization
python -m gradscope

# Or after installation
gradscope
```

## Usage

```python
from gradscope import MiniTransformer, TransformerVisualizer
import tkinter as tk

# Create and train a model programmatically
model = MiniTransformer(vocab_size=10, d_model=16, n_heads=2)

# Or launch the GUI
root = tk.Tk()
app = TransformerVisualizer(root)
root.mainloop()
```

## Project Structure

```
gradscope/
├── src/
│   └── gradscope/
│       ├── __init__.py
│       ├── model.py          # Transformer implementation
│       ├── visualization.py  # Tkinter GUI
│       └── main.py          # Entry point
├── tests/
│   ├── test_model.py
│   └── test_visualization.py
├── examples/
│   └── basic_usage.py
├── docs/
│   └── architecture.md
└── requirements.txt
```

## How It Works

The tool demonstrates:
1. **Forward Pass**: Input → Embedding → Attention → Output
2. **Loss Computation**: Cross-entropy loss for sequence prediction
3. **Backward Pass**: Gradients flow from loss through all layers
4. **Weight Updates**: Gradient descent updates model parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by "Attention Is All You Need" paper
- Built with NumPy and Tkinter for educational purposes
