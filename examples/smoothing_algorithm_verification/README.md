# Smoothing Algorithm Verification Tests

This directory contains a comprehensive test suite for verifying and comparing the different smoothing algorithms implemented in the VECTOR package.

## Test Suite Structure

```
smoothing_algorithm_verification/
├── README.md                 # This file
├── run_tests.py             # Main test runner script
├── test_config.py           # Test configuration parameters
└── test_cases/              # Individual test implementations
    ├── test_linear.py       # 2D Linear algorithm tests
    ├── test_3dlinear.py     # 3D Linear algorithm tests
    ├── test_vertex.py       # 2D Vertex algorithm tests
    ├── test_3dvertex.py     # 3D Vertex algorithm tests
    ├── test_levelset.py     # 2D Level Set algorithm tests
    └── test_3dlevelset.py   # 3D Level Set algorithm tests
```

## Algorithms Tested

1. Linear Smoothing (2D and 3D)
2. Vertex Method (2D and 3D)
3. Level Set Method (2D and 3D)

Each algorithm is tested for:
- Normal vector calculation accuracy
- Curvature calculation accuracy
- Computational performance
- Convergence behavior

## Test Cases

1. Circle/Sphere Tests
   - Tests accuracy on a simple geometry with known analytical solution
   - Validates normal vectors and curvature calculations
   - Measures computational performance

2. Voronoi Tests
   - Tests behavior on complex grain structures
   - Evaluates performance at triple junctions
   - Assesses robustness with multiple grains

3. Convergence Tests
   - Analyzes convergence rate with increasing iterations
   - Measures computational cost scaling
   - Determines optimal iteration counts

## Running the Tests

1. Basic Usage:
   ```bash
   python run_tests.py
   ```

2. Configuration:
   - Edit `test_config.py` to modify test parameters
   - Parameters include grid sizes, number of grains, iteration counts, etc.
   - Visualization settings can also be customized

## Output

The test suite generates:
1. Detailed test results in the `output/` directory
2. Comparative visualizations of algorithm performance
3. Comprehensive report comparing all methods
4. Performance metrics and error analysis

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- VECTOR package and its dependencies

## Customization

You can customize the tests by:
1. Modifying parameters in `test_config.py`
2. Adding new test cases in the `test_cases/` directory
3. Extending the comparison metrics in `run_tests.py`

## Result Analysis

The test suite compares the algorithms based on:
1. Accuracy
   - RMS error in normal vector calculations
   - RMS error in curvature calculations
   - Maximum errors and error distributions

2. Performance
   - Computation time
   - Memory usage
   - Scaling with problem size

3. Robustness
   - Behavior at triple junctions
   - Stability across different geometries
   - Convergence properties

## Contributing

To add new tests:
1. Create a new test file in `test_cases/`
2. Add corresponding configuration in `test_config.py`
3. Update `run_tests.py` to include the new test
4. Document the test cases and expected results