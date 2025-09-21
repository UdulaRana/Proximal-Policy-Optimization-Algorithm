# Contributing to PPO Algorithm Implementation

Thank you for your interest in contributing to the Proximal Policy Optimization Algorithm implementation! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project follows a Code of Conduct to ensure a welcoming environment for all contributors. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Git
- Basic understanding of reinforcement learning concepts
- Familiarity with PyTorch

### Areas for Contribution

We welcome contributions in several areas:

1. **Algorithm Implementation**: Core PPO algorithm components
2. **Environment Support**: New environment wrappers and integrations
3. **Documentation**: Improvements to documentation and examples
4. **Testing**: Unit tests, integration tests, and benchmarks
5. **Performance**: Optimization and efficiency improvements
6. **Features**: New features like different PPO variants

## Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/yourusername/Proximal-Policy-Optimization-Algorithm.git
   cd Proximal-Policy-Optimization-Algorithm
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv ppo_env
   source ppo_env/bin/activate  # On Windows: ppo_env\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e .[dev]
   ```

5. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Making Changes

### Branch Naming Convention

Use descriptive branch names:
- `feature/add-continuous-actions` - for new features
- `fix/memory-leak-training` - for bug fixes
- `docs/api-reference` - for documentation
- `test/actor-critic-networks` - for tests
- `refactor/training-loop` - for refactoring

### Commit Message Guidelines

Follow conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or modifying tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes

Examples:
```
feat(ppo): add support for continuous action spaces

fix(training): resolve memory leak in experience buffer

docs(readme): update installation instructions
```

### Code Organization

```
ppo/
├── core/
│   ├── agent.py          # Main PPO agent implementation
│   ├── networks.py       # Actor and Critic networks
│   ├── buffer.py         # Experience replay buffer
│   └── utils.py          # Utility functions
├── environments/
│   ├── wrappers.py       # Environment wrappers
│   └── multi_env.py      # Multi-environment support
├── training/
│   ├── trainer.py        # Training loop implementation
│   └── callbacks.py      # Training callbacks
├── evaluation/
│   ├── evaluator.py      # Evaluation utilities
│   └── metrics.py        # Performance metrics
└── config/
    ├── default.py        # Default configurations
    └── hyperparameters.py # Hyperparameter settings
```

## Testing

### Running Tests

Run the full test suite:
```bash
pytest tests/
```

Run specific test categories:
```bash
pytest tests/test_agent.py -v
pytest tests/test_networks.py -v
pytest tests/test_training.py -v
```

Run tests with coverage:
```bash
pytest tests/ --cov=ppo --cov-report=html
```

### Writing Tests

- Write unit tests for new functions and classes
- Include integration tests for complex workflows
- Test edge cases and error conditions
- Use descriptive test names

Example test structure:
```python
import pytest
import torch
from ppo.core.agent import PPOAgent

class TestPPOAgent:
    def test_initialization(self):
        """Test agent initialization with valid parameters."""
        agent = PPOAgent(state_dim=4, action_dim=2)
        assert agent.state_dim == 4
        assert agent.action_dim == 2
    
    def test_get_action_shape(self):
        """Test action output shape is correct."""
        agent = PPOAgent(state_dim=4, action_dim=2)
        state = torch.randn(1, 4)
        action = agent.get_action(state)
        assert action.shape == (1,)
```

### Benchmarking

When making performance changes, include benchmarks:

```python
import time
import numpy as np

def benchmark_training_step():
    """Benchmark training step performance."""
    agent = PPOAgent(state_dim=100, action_dim=10)
    states = torch.randn(1000, 100)
    
    start_time = time.time()
    for _ in range(100):
        actions = agent.get_action(states)
    end_time = time.time()
    
    print(f"Average time per batch: {(end_time - start_time) / 100:.4f}s")
```

## Code Style

### Python Code Style

We use the following tools for code formatting and linting:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run formatting tools:
```bash
black ppo/ tests/
isort ppo/ tests/
flake8 ppo/ tests/
mypy ppo/
```

### Style Guidelines

1. **Follow PEP 8** for Python code style
2. **Use type hints** for function parameters and return values
3. **Write docstrings** for all public functions and classes
4. **Keep functions focused** and limit complexity
5. **Use meaningful variable names**

Example of well-formatted code:
```python
from typing import Tuple, Optional
import torch
import torch.nn as nn

class ActorNetwork(nn.Module):
    """Actor network for policy approximation.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension
        activation: Activation function name
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64,
        activation: str = "tanh"
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        self.layers = self._build_network(hidden_dim, activation)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Action probabilities
        """
        return self.layers(state)
```

### Documentation Style

- Use **Google-style docstrings**
- Include **type information** in docstrings
- Provide **examples** for complex functions
- Keep documentation **up-to-date** with code changes

## Submitting Changes

### Pull Request Process

1. **Update your fork** with the latest changes:
   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** and commit them:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on GitHub

### Pull Request Guidelines

- **Provide a clear description** of the changes
- **Reference related issues** using `#issue-number`
- **Include tests** for new functionality
- **Update documentation** as needed
- **Ensure all checks pass** (tests, linting, etc.)

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
```

## Reporting Issues

### Bug Reports

When reporting bugs, include:

1. **Environment information**:
   - Python version
   - PyTorch version
   - Operating system
   - Hardware specifications

2. **Steps to reproduce**:
   - Minimal code example
   - Expected behavior
   - Actual behavior

3. **Additional context**:
   - Error messages
   - Stack traces
   - Screenshots (if applicable)

### Feature Requests

For feature requests, provide:

1. **Clear description** of the proposed feature
2. **Use case** and motivation
3. **Possible implementation** approach
4. **Alternatives considered**

### Issue Template

```markdown
## Bug Report / Feature Request

### Description
Clear description of the issue or feature

### Environment
- Python version:
- PyTorch version:
- OS:
- Hardware:

### Steps to Reproduce (for bugs)
1. Step 1
2. Step 2
3. Step 3

### Expected Behavior
What should happen

### Actual Behavior
What actually happens

### Additional Context
Any additional information
```

## Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- Project documentation

Thank you for contributing to the PPO Algorithm implementation!