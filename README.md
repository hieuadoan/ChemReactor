# ChemReactor

## 🧪 Python-Based Chemical Engineering Reactor Simulation Package

ChemReactor is a powerful, intuitive Python package for modeling, simulating, and visualizing chemical reactor systems. Built specifically for chemical engineers, researchers, and engineering students, it provides tools to explore reaction engineering concepts with interactive visualizations and accurate numerical simulations.

The package focuses on reactor design principles with a highly visual, educational approach that makes chemical reaction engineering more accessible and intuitive for practitioners and students alike.

## ✨ Features

- **Multiple Reactor Types**: Simulate batch reactors, continuous stirred-tank reactors (CSTRs), and plug flow reactors (PFRs)
- **Flexible Reaction Modeling**: Define simple or complex reaction networks with customizable kinetics
- **Interactive Visualizations**: See reactions in action with dynamic animations showing species concentrations
- **Robust Numerical Methods**: Powered by scientific Python libraries for accurate solutions to complex ODEs
- **Educational Focus**: Designed specifically for learning and teaching reactor engineering concepts
- **Modern Python Interface**: Clean, intuitive API that follows current Python best practices
- **Low Barrier to Entry**: Approachable for students and engineers new to Python or reaction modeling
- **Extensible Architecture**: Easily add new reactor types, kinetic models, or visualization methods

## 🚀 Installation (Pip installation in progress)

```bash
# Create new conda environment 
conda env create -f chemreactor.yml

# Activate environment 
conda activate chemreactor

# Install package in the development mode
pip install -e .
```

## 🔍 Quick Example

```python
from chemreactor.core.base_reactor import Reaction
from chemreactor.reactors.batch_reactor import BatchReactor
from chemreactor.visualization.reactor_animation import ReactorAnimation
import numpy as np
import matplotlib.pyplot as plt

# Define a reaction: A -> B
reaction = Reaction(
    reactants={'A': 1.0},
    products={'B': 1.0},
    rate_constant=0.1  # min^-1
)

# Initialize a batch reactor
reactor = BatchReactor(
    initial_concentrations={'A': 1.0, 'B': 0.0},
    reactions=[reaction]
)

# Run the simulation
results = reactor.run(end_time=60.0)

# Visualize the results
animator = ReactorAnimation(reactor)
animation = animator.create_batch_animation(results)
plt.show()
```

## 📚 Documentation

Visit our [documentation site](https://chemreactor.readthedocs.io/) for:
- Detailed API reference
- Tutorials and examples
- Theoretical background
- Advanced usage guides

## 🔧 Use Cases

- **Education**: Teach reaction engineering principles with interactive visualizations
- **Research**: Rapidly prototype and test reactor designs
- **Industry**: Model production reactors for optimization and troubleshooting
- **Process Development**: Simulate multi-step reaction processes
- **Visual Learning**: Create intuitive visualizations to enhance understanding
- **Engineering Design**: Bridge the gap between theoretical concepts and practical design

## 🛠️ Extending ChemReactor

The modular architecture makes it easy to extend with:
- Custom reactor types
- Specialized kinetic models
- New visualization techniques
- Integration with other chemical engineering tools

## 📊 Planned Features

- Thermodynamic calculations integration
- Transport phenomena modeling
- Catalytic reactor models
- Multiphase reactor support
- Heat transfer effects
- Reaction networks optimizer
- Interactive educational modules and tutorials
- Web-based visualization capabilities
- Industry-relevant reactor configurations
- Process optimization tools

## 🤝 Contributing

is always welcome. 

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔄 Key Advantages

ChemReactor offers several unique advantages for chemical engineering applications:

- **Designed by and for Chemical Engineers**: Built with the specific needs of ChemE practitioners in mind
- **Visualization-First Approach**: Prioritizes interactive visualizations for enhanced understanding
- **Educational Foundation**: Perfect for classroom use and self-learning
- **Engineering Design Focus**: Emphasizes practical reactor design principles
- **Pure Python Implementation**: Easy to customize and extend for specific engineering problems
- **Low Barrier to Entry**: Accessible to engineering students and professionals at all coding levels

---

*ChemReactor: Bringing chemical reaction engineering to life with Python.*
