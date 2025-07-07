# Grain Boundary Velocity and Anti-Curvature Analysis Documentation

## Overview

This documentation describes a comprehensive suite of Jupyter notebooks for analyzing grain boundary (GB) velocity and anti-curvature behavior in polycrystalline materials. The analysis covers both 2D and 3D systems with different anisotropy types, providing insights into how grain boundary energy anisotropy affects grain growth dynamics.

## 🔬 Scientific Background

**Anti-curvature behavior** occurs when grain boundaries move opposite to the direction predicted by their curvature drive force. In normal grain growth, high-curvature boundaries should shrink and low-curvature boundaries should grow. However, in anisotropic systems, energy landscape variations can cause boundaries to exhibit counter-intuitive motion patterns.

**Key Anisotropy Types:**
- **Isotropic (iso)**: No directional energy dependence 
- **Misorientation-only (M)**: Energy depends only on misorientation angle (f=1.0, t=0.0)
- **Inclination-only (I)**: Energy depends only on boundary plane orientation (f=0.0, t=1.0)  
- **Full Anisotropy (MI)**: Energy depends on both misorientation and inclination

**Energy Function Types:**
- **Standard (no suffix)**: Uses **Cosine energy function** for grain boundary energy calculation
- **Well Energy variants (_ab)**: Uses **Well Energy function** for grain boundary energy calculation

---

## 📁 Notebook Categories

### 2D Analysis Notebooks

#### Core Analysis Suite
| Notebook | Anisotropy Type | Description |
|----------|----------------|-------------|
| `GB_velocity_anti_curvature-iso.ipynb` | Isotropic | Baseline analysis for isotropic grain boundary energy |
| `GB_velocity_anti_curvature_M.ipynb` | Misorientation-only | Focuses on misorientation angle effects (t=0.0) |
| `GB_velocity_anti_curvature_I.ipynb` | Inclination-only | Focuses on boundary plane orientation effects (f=0.0) |
| `GB_velocity_anti_curvature_MI.ipynb` | Full Anisotropy | Combined misorientation and inclination effects |

#### Enhanced Variants  
| Notebook | Anisotropy Type | Description |
|----------|----------------|-------------|
| `GB_velocity_anti_curvature_M_ab.ipynb` | M + Well Energy | Misorientation analysis with Well Energy function |
| `GB_velocity_anti_curvature_I_ab.ipynb` | I + Well Energy | Inclination analysis with Well Energy function |
| `GB_velocity_anti_curvature_MI_ab.ipynb` | MI + Well Energy | Full anisotropy analysis with Well Energy function |

### 3D Analysis Notebooks

#### Core 3D Analysis
| Notebook | Anisotropy Type | Description |
|----------|----------------|-------------|
| `3D_GB_velocity_anti_curvature_iso.ipynb` | Isotropic | 3D baseline analysis |
| `3D_GB_velocity_anti_curvature_M.ipynb` | Misorientation-only | 3D misorientation effects |
| `3D_GB_velocity_anti_curvature_I.ipynb` | Inclination-only | 3D inclination effects |
| `3D_GB_velocity_anti_curvature_MI.ipynb` | Full Anisotropy | 3D combined effects |

#### Enhanced 3D Variants
| Notebook | Anisotropy Type | Description |
|----------|----------------|-------------|
| `3D_GB_velocity_anti_curvature_M_Ab.ipynb` | M + Well Energy | 3D misorientation analysis with Well Energy function |
| `3D_GB_velocity_anti_curvature_I_Ab.ipynb` | I + Well Energy | 3D inclination analysis with Well Energy function |
| `3D_GB_velocity_anti_curvature_MI_Ab.ipynb` | MI + Well Energy | 3D full anisotropy analysis with Well Energy function |

#### Specialized 3D Analysis
| Notebook | Purpose | Description |
|----------|---------|-------------|
| `3D_GB_velocity_anti_curvature_steady_state.ipynb` | Steady-state | Analysis of grain growth in steady-state regime |
| `3D_GB_velocity_anti_curvature_steady_state_experiment.ipynb` | Experimental | Experimental validation of steady-state analysis |
| `3D_GB_velocity_anti_curvature_iso_VECTOR.ipynb` | VECTOR Method | Specialized VECTOR algorithm implementation |
| `3D_GB_velocity_anti_curvature_MI_VECTOR.ipynb` | VECTOR + MI | VECTOR algorithm with full anisotropy |

### Comparative Analysis Notebooks

#### Statistical Analysis Suite
| Notebook | Purpose | Description |
|----------|---------|-------------|
| `anti_curvature_fraction_plot.ipynb` | Fraction Comparison | Comparative anti-curvature fraction analysis across anisotropy types and energy functions |

#### Algorithm Validation Notebooks
| Notebook | Purpose | Description |
|----------|---------|-------------|
| `IO_curvature_calculate_for_oneGB.ipynb` | Algorithm Validation | Single grain boundary curvature calculation validation for 2D and 3D isotropic systems |

---

## 🧩 Functional Block Structure

All notebooks follow a consistent block-level structure for systematic analysis:

### Block 1: Environment Setup and Data Loading
**Purpose**: Initialize analysis environment and load simulation data

**Key Components**:
- **Library imports**: NumPy, matplotlib, tqdm, multiprocessing
- **Path management**: Custom module imports (VECTOR packages, post_processing)
- **Data file specification**: SPPARKS simulation results (microstructure + energy)
- **Parameter configuration**: Case names, file paths, analysis parameters

**Example Variables**:
```python
case_name = "MI_20k_iso"  # Analysis case identifier
data_file_folder = "/path/to/VECTOR_data/"  # Results storage
npy_file_folder = "/path/to/SPPARKS/results/"  # SPPARKS data
step_num, size_x, size_y, size_z = npy_file_aniso.shape  # Data dimensions
```

### Block 2: Grain Boundary Curvature Calculation  
**Purpose**: Extract and calculate signed curvature for all grain boundaries

**Key Components**:
- **GB detection**: Identify interface sites between different grains
- **Curvature computation**: Use linear smoothing algorithms for signed curvature
- **Triple junction handling**: Remove sites near triple/quadruple junctions
- **Data caching**: Store computed curvature data for reuse

**Analysis Methods**:
- **2D**: Linear smoothing with 5x5 kernel matrices
- **3D**: Linear smoothing with 5x5x5 kernel matrices  
- **Quality filtering**: Area thresholds, distance from junctions

### Block 3: Grain Boundary Energy Extraction
**Purpose**: Calculate energy per unit length for each grain boundary

**Key Components**:
- **Energy site identification**: Map energy data to GB locations
- **Junction filtering**: Remove triple/quadruple junction influences  
- **Energy normalization**: Calculate energy per neighboring grain
- **Statistical aggregation**: Average energy over GB length/area

**Anisotropy-Specific Features**:
- **Isotropic**: Constant energy per unit length
- **Misorientation (M)**: Energy varies with misorientation angle
- **Inclination (I)**: Energy varies with boundary plane orientation
- **Full (MI)**: Energy varies with both parameters

**Energy Function Differences**:
- **Cosine Energy Function (standard)**: Smooth, continuous energy variation with crystallographic parameters
- **Well Energy Function (_ab variants)**: Discrete energy wells at specific crystallographic configurations, creating sharper energy minima

### Block 4: Velocity Calculation Functions
**Purpose**: Define optimized functions for GB velocity computation

**Key Components**:
- **Volume change tracking**: Count grain switching between time steps
- **Numba optimization**: JIT compilation for performance (`@njit(parallel=True)`)
- **Directional analysis**: Separate tracking of growth directions
- **Anti-curvature detection**: Identify velocity-curvature sign mismatch

**Critical Functions**:
```python
@njit(parallel=True)
def compute_dV(current, next, pair_ids):
    # Calculate net volume change for grain boundary
    
@njit(parallel=True)  
def compute_dV_split(current, next, pair_ids):
    # Calculate volume change with directional tracking
    
def compute_necessary_info_split(key, time_interval, GB_info, energy_info, current, next):
    # Complete velocity-curvature analysis for single GB
```

### Block 5: Main Velocity-Curvature Analysis Loop
**Purpose**: Process all time steps to extract velocity-curvature relationships

**Key Components**:
- **Quality filtering**: Remove small GBs, disappeared GBs, low curvature GBs
- **Parallel processing**: Multiprocessing for computational efficiency
- **Time series tracking**: Monitor GB evolution across time steps
- **Anti-curvature identification**: Flag GBs with velocity × curvature < 0

**Analysis Parameters**:
```python
time_interval = 30        # Time step interval for velocity calculation
curvature_limit = 0.0182  # Minimum curvature threshold
area_limit = 100          # Minimum area threshold
```

### Block 6: Temporal Pattern Filtering
**Purpose**: Identify persistent vs. transient anti-curvature behavior

**Key Components**:
- **Rolling window analysis**: Track anti-curvature patterns across 5 time steps
- **Pattern classification**: Different temporal signatures (00100, 11111, etc.)
- **Statistical filtering**: Remove noise and focus on significant events
- **GB lifecycle tracking**: Monitor individual GB behavior over time

**Filter Patterns**:
- **00100**: Isolated anti-curvature event (single time step)
- **11111**: Persistent anti-curvature (all time steps)
- **01100/00110**: Consecutive anti-curvature events

### Block 7: Statistical Analysis and Data Processing
**Purpose**: Aggregate and analyze velocity-curvature correlations

**Key Components**:
- **Data aggregation**: Combine all time steps into master lists
- **Correlation analysis**: Velocity vs. curvature relationships
- **Energy landscape analysis**: Energy vs. velocity correlations
- **Statistical binning**: Curvature-based histogram analysis

**Output Data Structures**:
```python
GB_list_velocity_list = []      # All GB velocities
GB_list_curvature_list = []     # All GB curvatures  
GB_list_GBenergy_list = []      # All GB energies
GB_list_area_list = []          # All GB areas
GB_id_focus = {}                # Anti-curvature GB tracking
```

### Block 8: Visualization and Results
**Purpose**: Generate comprehensive plots and analysis figures

**Key Components**:
- **Scatter plots**: Velocity vs. curvature relationships
- **Density plots**: High-density data visualization with binning
- **Histogram analysis**: Statistical distributions
- **Trend analysis**: Linear fits and correlation coefficients

**Typical Visualizations**:
- **Main scatter plot**: All GBs with anti-curvature highlighting
- **Binned analysis**: Average velocity vs. curvature in bins
- **Energy correlation**: Velocity vs. GB energy relationships
- **Temporal evolution**: Anti-curvature fraction over time

**3D-Specific Additions**:
- **Mean curvature analysis**: 3D curvature calculations
- **Direction-dependent effects**: Anisotropy in 3D space
- **Volume analysis**: 3D grain volume evolution

### Block 9: Advanced Analysis (Specialized Notebooks)
**Purpose**: Extended analysis for specific research questions

**Steady-State Analysis**:
- **Grain size distribution**: Evolution toward steady state
- **Anti-curvature persistence**: Long-term behavior patterns
- **Statistical convergence**: Equilibrium velocity-curvature relationships

**VECTOR Method Analysis**:
- **Algorithm comparison**: VECTOR vs. traditional methods
- **Computational efficiency**: Performance benchmarking
- **Accuracy validation**: Error analysis and convergence

---

## 🔄 Workflow Integration

### Sequential Analysis Workflow
1. **Algorithm validation** (`IO_curvature_calculate_for_oneGB.ipynb`) to verify curvature calculation methods on simple systems
2. **Start with isotropic** (`*-iso.ipynb`) to establish baseline behavior
3. **Progress to single anisotropy** (`*_M.ipynb` or `*_I.ipynb`) to isolate effects
4. **Analyze full anisotropy** (`*_MI.ipynb`) to understand combined effects
5. **Compare energy functions** (`*_ab.ipynb`) to examine Well Energy vs. Cosine Energy effects

### Comparative Studies
- **2D vs 3D**: Compare dimensionality effects on anti-curvature behavior
- **Anisotropy types**: Understand relative importance of misorientation vs. inclination
- **Energy functions**: Compare Cosine Energy vs. Well Energy function impacts on grain boundary dynamics
- **Steady-state**: Examine long-term vs. transient behavior
- **VECTOR validation**: Verify algorithm accuracy and efficiency
- **Fraction analysis**: Use `anti_curvature_fraction_plot.ipynb` for comprehensive cross-system comparison of anti-curvature prevalence
- **Energy binning**: Use `anti-c_fraction_bin.ipynb` for detailed energy-dependent correlation analysis

### Meta-Analysis Workflow  
1. **Algorithm validation**: Use `IO_curvature_calculate_for_oneGB.ipynb` to verify curvature calculation accuracy on simple two-grain systems
2. **Individual system analysis**: Complete analysis for each anisotropy type and energy function
3. **Data aggregation**: Collect anti-curvature fractions and statistics from all systems
4. **Comparative visualization**: Use comparative notebooks to identify trends and patterns
5. **Energy correlation**: Analyze energy-dependent anti-curvature behavior across systems
6. **Publication synthesis**: Generate comprehensive figures and statistical summaries

### Algorithm Validation Workflow
1. **Simple system testing**: Two-grain isotropic systems for clean validation environment
2. **Method comparison**: Direct comparison of 2D (linear smoothing) vs 3D (IO_curvature) algorithms
3. **Temporal consistency**: Verify curvature evolution patterns match physical expectations
4. **Reference generation**: Create benchmark data for validating complex system analyses

### Data Pipeline
```
SPPARKS Simulation → Microstructure Data → Curvature Calculation → 
Energy Extraction → Velocity Analysis → Anti-curvature Detection → 
Pattern Analysis → Visualization → Scientific Interpretation
```

---

## 🎯 Key Scientific Insights

### Anti-Curvature Phenomena
- **Physical mechanism**: Energy anisotropy creates local energy minima that drive anti-curvature motion
- **Frequency**: Typically 1-5% of grain boundaries show persistent anti-curvature behavior
- **Anisotropy dependence**: More prevalent in systems with strong inclination anisotropy
- **Energy function sensitivity**: Well Energy functions can enhance anti-curvature behavior due to sharper energy gradients compared to Cosine Energy functions
- **Dimensional effects**: 3D systems often show higher absolute anti-curvature fractions but different energy-dependent patterns compared to 2D systems

### Comparative Analysis Insights
- **Energy function ranking**: Well Energy functions generally produce different anti-curvature patterns compared to Cosine Energy functions, with case-specific variations
- **Anisotropy susceptibility**: Inclination-dependent (I) and Full (MI) anisotropy types typically show higher anti-curvature fractions than Misorientation-only (M) systems
- **Statistical confidence**: High-confidence thresholds (99%) significantly reduce apparent anti-curvature fractions, emphasizing the importance of statistical rigor
- **Cross-system validation**: Multiple analysis methodologies (original vs. normalized) provide validation of anti-curvature detection algorithms

### Computational Methods
- **Curvature calculation**: Linear smoothing provides accurate signed curvature
- **Velocity tracking**: Volume change method captures grain boundary motion
- **Pattern detection**: Temporal filtering identifies meaningful anti-curvature events
- **Energy function handling**: Both Cosine and Well Energy functions integrated in analysis framework
- **Algorithm validation**: Single GB systems provide clean validation environment for curvature calculation methods
- **2D vs 3D methods**: Linear smoothing (2D) vs IO_curvature (3D) algorithms validated on equivalent systems

### Material Applications
- **Microstructure control**: Understanding when grain growth deviates from predictions
- **Texture development**: Anti-curvature affects crystallographic texture evolution
- **Property optimization**: Controlling grain boundary character distribution
- **Energy function selection**: Choice between Cosine and Well Energy functions affects predicted microstructure evolution

---

## 🛠 Technical Requirements

### Dependencies
- **Core Python**: NumPy, matplotlib, SciPy
- **Performance**: Numba (JIT compilation), multiprocessing
- **VECTOR packages**: Custom linear solvers and post-processing tools
- **Data formats**: NPY files from SPPARKS simulations

### Computational Resources
- **Memory**: 16-32 GB RAM for typical 20k grain simulations
- **Processing**: Multi-core CPU for parallel velocity calculations
- **Storage**: Several GB for cached intermediate results

### Input Data Format
- **Microstructure files**: `.npy` format with grain IDs over time
- **Energy files**: `.npy` format with corresponding energy data
- **Dimensions**: 2D (time × x × y × 1) or 3D (time × x × y × z)

This comprehensive analysis suite provides a complete framework for understanding grain boundary dynamics in anisotropic materials, enabling both fundamental research and practical materials engineering applications.
