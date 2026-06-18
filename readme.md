# Heetei Recon
## Overview & Context
This repository features a pipeline for 3D cave reconstruction using Lidar data and Alpha Shapes/Ball Pivoting algorithms. It processes raw `.laz` point clouds to reconstruct complex subterranean geometries 

## Tech Stack & Keywords
* **Core Libraries:** `open3d==0.19.0`, `laspy[lazrs]==2.5.4`, `numpy==1.26.4`, `pyyaml==6.0.2`
* **Memory Optimization:** Custom NumPy pipeline targeting `float32` instead of default `float64` to dramatically reduce RAM footprint on resource-constrained Edge devices.
* **Keywords:** 3D Reconstruction, Lidar Processing, `.laz` Point Clouds, Alpha Shapes, Ball Pivoting Algorithm, Memory Efficiency.

## Core Architecture
The project follows a modular pipeline structure separated into logical stages:
* `main.py` — Entry point that orchestrates the whole pipeline using parameters from `config.yaml`.
* `utils/ingest.py` — Handles `.laz` file reading and memory-efficient data conversion.
* `pipeline/preprocess.py` — Responsible for data filtering and downsampling.
* `pipeline/reconstruct.py` — Implements surface reconstruction methods (Alpha Shapes / Ball Pivoting).
* `pipeline/visualize.py` — Manages 3D rendering, mesh visualization and saving using Open3D.

## How to Run
### Recommended (Google Colab)
Run in Google Colab:
```bash
!git clone https://github.com/Shamaniks/heetei-recon
!mv heetei-recon/* ./
!pip install -r requirements.txt # And restart kernel if asked
# Download your .laz file
# Set your config in config.yaml as needed
!python main.py --file <file>
```

### Local Setup (Linux/macOS)
```bash
git clone https://github.com/Shamaniks/heetei-recon
cd heetei-recon
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# For reconstruction only
python main.py --file <file>

# For Open3D visualization
DISPLAY=:0 python main.py --file <file>
```

## Future Roadmap
- [ ] Analyze point intensity and implement filtering for smoke, steam, and dust
- [ ] Implement compatibility with Therion (speleological survey software)
