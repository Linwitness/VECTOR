# VECTOR
VoxEl-based boundary inClination smooThing AlgORithms

This project provides four smoothing algorithms to get the inclination of each voxel on the 2D and 3D sharp boundary: Vertex smoothing algorithm, Allen-Cahn smoothing algorithm, Level-Set smoothing algorithm and Bilinear smoothing algorithm.

All smoothing algorithms can work in parallel and are integrated with comprehensive analysis tools for materials science applications, particularly grain growth simulation and grain boundary characterization.

## Project Structure Overview

```
VECTOR/
├── Core Algorithm Files
│   ├── main.py                          # Main execution script for smoothing algorithms
│   ├── myInput.py                       # Input utilities and data processing functions
│   ├── post_processing.py               # Post-processing and analysis utilities
│   └── PACKAGE_MP_*.py                  # Core smoothing algorithm implementations
├── Algorithm Packages
│   ├── PACKAGE_MP_Vertex.py             # 2D Vertex smoothing algorithm
│   ├── PACKAGE_MP_Linear.py             # 2D Bilinear smoothing algorithm
│   ├── PACKAGE_MP_AllenCahn.py          # 2D Allen-Cahn smoothing algorithm
│   ├── PACKAGE_MP_LevelSet.py           # 2D Level-Set smoothing algorithm
│   ├── PACKAGE_MP_3DVertex.py           # 3D Vertex smoothing algorithm
│   ├── PACKAGE_MP_3DLinear.py           # 3D Bilinear smoothing algorithm
│   ├── PACKAGE_MP_3DAllenCahn.py        # 3D Allen-Cahn smoothing algorithm
│   └── PACKAGE_MP_3DLevelSet.py         # 3D Level-Set smoothing algorithm
├── Data Directories
│   ├── input/                           # Input examples and test cases
│   └── output/                          # Algorithm results and output data
├── Application Examples
│   ├── examples/                        # Comprehensive application examples
│   └── verification/                    # Algorithm verification and testing
└── Documentation and Metadata
    ├── README.md                        # This documentation
    ├── CITATION.cff                     # Citation information
    └── __init__.py                      # Python package initialization
```

## Core Algorithm Files

### main.py
**Purpose**: Main execution script for VECTOR smoothing algorithms
- **Primary Function**: Entry point for running all four smoothing algorithms (Vertex, Linear, Allen-Cahn, Level-Set)
- **Key Features**: 
  - Unified interface for 2D and 3D algorithm execution
  - Parameter configuration and algorithm selection
  - Parallel processing coordination
  - Output management and result validation

### myInput.py (993 lines)
**Purpose**: Comprehensive input utilities and data processing functions
- **Core Functions**:
  - Input file processing from various formats (SPPARKS, DREAM.3D, custom formats)
  - Microstructure data handling and conversion utilities
  - Periodic boundary condition management
  - Geometric utility functions for 2D and 3D systems
  - Data validation and preprocessing tools
- **Key Applications**: Interface between simulation data and VECTOR algorithms

### post_processing.py (1536 lines)
**Purpose**: Extensive post-processing and analysis utilities
- **Core Functions**:
  - Grain boundary analysis and characterization
  - Curvature calculation and validation
  - Statistical analysis tools for microstructure evolution
  - Visualization utilities for 2D and 3D data
  - Integration with SPPARKS simulation framework
  - Energy function analysis and grain growth metrics
- **Key Applications**: Scientific analysis and result interpretation

## Algorithm Package Details

### 2D Smoothing Algorithms

#### PACKAGE_MP_Vertex.py
- **Algorithm**: Vertex-based boundary smoothing for 2D systems
- **Key Features**: Vertex identification, geometric smoothing, parallel processing
- **Applications**: Sharp boundary smoothing with vertex preservation

#### PACKAGE_MP_Linear.py
- **Algorithm**: Bilinear smoothing for 2D grain boundary inclination calculation
- **Key Features**: Linear interpolation, gradient calculation, parallel implementation
- **Applications**: Grain boundary normal vector calculation and inclination analysis

#### PACKAGE_MP_AllenCahn.py
- **Algorithm**: Allen-Cahn equation-based smoothing for 2D systems
- **Key Features**: Phase field methodology, diffuse interface modeling
- **Applications**: Smooth boundary evolution and interface characterization

#### PACKAGE_MP_LevelSet.py
- **Algorithm**: Level-Set method for 2D boundary smoothing
- **Key Features**: Level-set function evolution, interface tracking
- **Applications**: Complex topology handling and boundary evolution

### 3D Smoothing Algorithms

#### PACKAGE_MP_3DVertex.py
- **Algorithm**: 3D vertex-based boundary smoothing
- **Key Features**: 3D vertex identification, volumetric smoothing, enhanced parallel processing
- **Applications**: Complex 3D grain boundary analysis and smoothing

#### PACKAGE_MP_3DLinear.py
- **Algorithm**: 3D bilinear smoothing for grain boundary analysis
- **Key Features**: 3D gradient calculation, mean curvature computation, parallel 3D processing
- **Applications**: 3D grain boundary characterization and curvature analysis

#### PACKAGE_MP_3DAllenCahn.py
- **Algorithm**: 3D Allen-Cahn equation-based smoothing
- **Key Features**: 3D phase field modeling, volumetric diffuse interfaces
- **Applications**: 3D microstructure evolution and complex interface analysis

#### PACKAGE_MP_3DLevelSet.py
- **Algorithm**: 3D Level-Set method for boundary smoothing
- **Key Features**: 3D level-set evolution, complex 3D topology handling
- **Applications**: Advanced 3D interface tracking and boundary evolution

## Examples Directory Structure

### examples/GB_velocity/
**Purpose**: Grain boundary velocity and anti-curvature analysis
- **Content**: 23 Jupyter notebooks for comprehensive grain boundary dynamics analysis
- **Key Features**: 2D/3D analysis, energy function comparison, anti-curvature detection
- **Applications**: Understanding counter-intuitive grain boundary motion patterns

### examples/verify_energy_function/
**Purpose**: Energy function verification through statistical analysis
- **Content**: 6 analysis tools (Python scripts and Jupyter notebooks)
- **Key Features**: Grain size distribution analysis, misorientation distribution characterization
- **Applications**: Energy function validation and comparative analysis

### examples/dump_to_init/
**Purpose**: SPPARKS preprocessing utilities
- **Content**: 5 specialized tools for simulation data processing
- **Key Features**: Microstructure generation, data conversion, neighbor connectivity
- **Applications**: Bridge between simulation preparation and execution

### examples/calculate_inclination/
**Purpose**: Grain boundary inclination calculation and validation
- **Content**: PRIMME-based inclination calculation tools
- **Key Features**: High-performance inclination computation, HiPerGator integration
- **Applications**: Large-scale crystallographic analysis

### examples/calculate_tangent/
**Purpose**: Tangent vector calculation for grain boundaries
- **Content**: Triple junction analysis and tangent vector computation
- **Key Features**: Dihedral angle calculation, tangent vector analysis
- **Applications**: Triple junction characterization and geometric analysis

### examples/curvature_calculation/
**Purpose**: Comprehensive curvature calculation and validation
- **Content**: Multiple curvature analysis tools and validation datasets
- **Key Features**: Algorithm comparison, geometric validation, 3D curvature analysis
- **Applications**: Curvature algorithm development and verification

### examples/microstructure/
**Purpose**: Microstructure visualization and analysis
- **Content**: Visualization tools for various microstructure types
- **Key Features**: 2D/3D plotting, statistical analysis, publication-quality figures
- **Applications**: Microstructure characterization and visual analysis

### examples/plot_GG_property/
**Purpose**: Grain growth property analysis and visualization
- **Content**: Comprehensive grain growth analysis tools
- **Key Features**: Statistical analysis, temporal evolution, energy analysis
- **Applications**: Grain growth kinetics and microstructure evolution studies

### examples/TJ_site_energy_calculation/
**Purpose**: Triple junction site energy calculation
- **Content**: Energy calculation tools for triple junction analysis
- **Key Features**: Site-specific energy analysis, junction characterization
- **Applications**: Triple junction energy analysis and materials design

### examples/get_normals_TJangles/
**Purpose**: Normal vector and triple junction angle analysis
- **Content**: Geometric analysis tools for complex junction systems
- **Key Features**: Normal vector calculation, angle analysis
- **Applications**: Advanced geometric characterization of grain boundaries

## Verification Framework

### verification/smoothing_algorithm_verification/
**Purpose**: Comprehensive algorithm verification and testing suite
- **Content**: 
  - `run_tests.py`: Automated testing framework
  - `test_config.py`: Test configuration management
  - `test_cases/`: Comprehensive test case library
  - `output/`: Verification results and benchmarks
- **Key Features**: Algorithm accuracy validation, performance benchmarking, regression testing
- **Applications**: Quality assurance and algorithm development validation

## Data Directories

### input/
**Purpose**: Input examples and test cases
- **Content**: Sample microstructures and configuration files
- **Applications**: Algorithm testing and example execution

### output/
**Purpose**: Algorithm results and output data
- **Content**: Generated results from smoothing algorithms and analysis tools
- **Applications**: Result storage and validation data

## Scientific Applications

### Materials Science Research
- **Grain Growth Studies**: Comprehensive analysis of grain growth kinetics and mechanisms
- **Microstructure Evolution**: Statistical characterization of microstructural development
- **Energy Function Development**: Validation and optimization of grain boundary energy models
- **Crystallographic Analysis**: Advanced texture and orientation analysis

### Computational Materials Science
- **Algorithm Development**: Development and validation of new smoothing algorithms
- **Simulation Integration**: Interface with Monte Carlo and phase field simulations
- **Performance Optimization**: Parallel processing and computational efficiency
- **Method Validation**: Comprehensive testing and verification frameworks

### Engineering Applications
- **Materials Design**: Microstructure-property relationship analysis
- **Processing Optimization**: Understanding processing-microstructure connections
- **Quality Control**: Microstructure characterization for manufacturing
- **Property Prediction**: Computational prediction of material properties

## Technical Specifications

### Computational Capabilities
- **Parallel Processing**: All algorithms support multicore parallel execution
- **Scalability**: Designed for large-scale 3D microstructures (up to 1000³ voxels)
- **Memory Efficiency**: Optimized data structures for large dataset processing
- **HPC Integration**: Compatible with high-performance computing environments

### Supported Data Formats
- **SPPARKS**: Native integration with SPPARKS simulation output
- **DREAM.3D**: Support for experimental microstructure data
- **HDF5**: High-performance scientific data format support
- **NumPy**: Efficient numerical array processing
- **Custom Formats**: Flexible input/output for various data types

### Algorithm Performance
- **2D Systems**: Optimized for high-resolution 2D microstructure analysis
- **3D Systems**: Advanced 3D algorithms with enhanced computational efficiency
- **Accuracy**: Validated against analytical solutions and experimental data
- **Robustness**: Comprehensive error handling and validation procedures

## Installation and Dependencies

### Core Dependencies
```bash
pip install numpy matplotlib scipy tqdm numba
pip install h5py multiprocess
```

### Optional Dependencies
```bash
pip install jupyter ipywidgets  # For interactive notebooks
pip install plotly  # For advanced 3D visualization
pip install torch  # For GPU-accelerated processing
```

### HPC Environment
- **SLURM Integration**: Compatible with SLURM job scheduling
- **Module System**: Works with HPC module environments
- **Large Memory Support**: Optimized for high-memory computational nodes

## Usage Examples

### Basic Algorithm Execution
```python
# Execute VECTOR smoothing analysis
python main.py --algorithm=Linear --dimension=3D --cores=32
```

### Interactive Analysis
```python
# Use Jupyter notebooks for interactive analysis
jupyter notebook examples/GB_velocity/3D_GB_velocity_analysis.ipynb
```

### Large-Scale HPC Analysis
```bash
# SLURM job submission for large-scale analysis
sbatch --nodes=1 --ntasks=64 --mem=256GB run_vector_analysis.sh
```

# Publication Reference
**Primary Citation**: https://www.sciencedirect.com/science/article/pii/S1359646222002925

# Author Contact Information
**Lin Yang**  
Email: lin.yang@ufl.edu/linyangjump@outlook.com
Address: 633 Gale Lemerand Dr, Gainesville, FL 32603 (University of Florida)

---

For detailed documentation of specific components, refer to the README.md files in individual example directories and the comprehensive inline documentation within each module.
