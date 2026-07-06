# Heetei Recon
## Overview & Context
This repository features a tools for 3D cave reconstruction using Lidar data and Alpha Shapes/Ball Pivoting algorithms. It processes raw `.laz` point clouds to reconstruct complex subterranean geometries 

## Tech Stack & Keywords
* **Core Libraries:** `open3d==0.19.0`, `laspy[lazrs]==2.5.4`, `numpy==1.26.4`, `pyyaml==6.0.2`
* **Memory Optimization:** Custom NumPy pipeline targeting `int32` + `float64` scale instead of default `float64` to dramatically reduce RAM footprint on resource-constrained Edge devices
* **Keywords:** 3D Reconstruction, Lidar Processing, `.laz` Point Clouds, Alpha Shapes, Ball Pivoting Algorithm, Memory Efficiency

## Core Architecture
The project follows a modular tools structure separated into python modules (call: `python -m <module> <args (read docs for each module)>:
* `utils.laz_to_npz` — Handles `.laz` file reading and memory-efficient data conversion
* `restore_intensity` — Responsible for restoring real intensity from written (intensity decreases per distance)
* `downsampling` - Responsible for downsampling (voxelizing) any xyz, handles with and without sorting by intensity modes
* `visualization.points` — Manages 3D rendering using Open3D

## How to Run
### Recommended (Google Colab)
Run in Google Colab:
```bash
!git clone https://github.com/Shamaniks/heetei-recon
!mv heetei-recon/* ./
!pip install -r requirements.txt # And restart kernel if asked
# Download your .laz file
# Set your config in config.yaml as needed
!python -m <module> <args>
```

### Local Setup (Linux/macOS)
```bash
git clone https://github.com/Shamaniks/heetei-recon
cd heetei-recon
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m <module> <args>
```

## Future Roadmap
- [ ] Implement direct reader for splitted Mandeye session chunks (`lidarXXXX.laz` + `imuXXXX.csv`) instead of united `.laz` after HDMapping
- [ ] Usage of scanning `gpstime` for finding source track point instead of finding closest via `scipy.spatial.cKDtree`
- [ ] Polars hashtable for voxelisation instead of np.unique for better time optimization
- [ ] Visualization module for point clouds with colors passed, mesh grid
- [ ] Intensity to viridis colors translation
- [ ] Ghost filtering
- [ ] Clustering for different surface types and formations (stalagmites, rocks, etc.)
- [ ] Compatibility with Therion (speleological survey software)
