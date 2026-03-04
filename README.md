# OptiXDenoiser for Nuke

A GPU-accelerated AI denoising plugin for [Foundry's Nuke](https://www.foundry.com/products/nuke), powered by NVIDIA's OptiX denoising technology. OptiXDenoiser brings real-time, production-quality noise reduction directly into your compositing pipeline — no external tools or round-tripping required.

![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia)
![OptiX](https://img.shields.io/badge/OptiX-9.0-76B900?logo=nvidia)

---

## Features

- **AI-Powered Denoising** — Uses NVIDIA OptiX's deep-learning denoiser trained on thousands of 3D-rendered scenes to intelligently remove noise and fireflies from CG renders.
- **Guide AOV Support** — Connect optional Albedo, Normal, and Motion Vector passes for significantly improved denoising quality.
- **Temporal Denoising** — When motion vectors are connected, the plugin automatically enables temporal mode for flicker-free, stable results across frame sequences.
- **Tiled Processing** — Configurable tile sizes (512×512, 1024×1024, 2048×2048) allow the plugin to work within your GPU's VRAM budget, even on large-resolution images.
- **Automatic Fallback** — If denoising fails for any reason, the plugin gracefully falls back to passing the input through unmodified, preventing pipeline crashes.
- **Cross-Platform** — Pre-compiled binaries provided for both Linux and Windows across multiple Nuke versions.

---

## Supported Nuke Versions

| Nuke Version | Linux | Windows |
|:-------------|:-----:|:-------:|
| 14.1         | ✅    | ✅      |
| 15.0         | ✅    | ✅      |
| 15.1         | ✅    | ✅      |
| 15.2         | ✅    | ✅      |
| 16.0         | ✅    | ✅      |
| 16.1         | ✅    | ✅      |
| 17.0         | ✅    | ✅      |

---

## Requirements

- **NVIDIA GPU** with Kepler architecture or newer
- **NVIDIA Driver** — version 465.84 or higher recommended
- **CUDA Runtime** — statically linked into the plugin (no separate CUDA install needed for end users)

> **Note:** OptiX denoising is not available on macOS.

---

## Installation

### Using Pre-compiled Binaries

1. Download the appropriate binary for your platform and Nuke version from the `LINUX/` or `WIN/` folder.
2. Copy the plugin file to your `~/.nuke` directory (or any path in your `NUKE_PATH`):
   - **Linux:** `OptiXDenoiser.so`
   - **Windows:** `OptiXDenoiser.dll`
3. Restart Nuke. The node will appear under **Filter → OptiXDenoiser**.

---

## Usage

### Node Inputs

The OptiXDenoiser node accepts up to four inputs:

| Input   | Required | Description |
|:--------|:--------:|:------------|
| `beauty` | ✅ | The noisy rendered image (RGB) |
| `albedo` | ❌ | Albedo/diffuse color AOV — improves detail preservation |
| `normal` | ❌ | World-space normals AOV — improves edge-aware denoising (requires albedo) |
| `motion` | ❌ | Motion vectors AOV — enables temporal denoising mode |

### Node Parameters

**Temporal Settings**
- **Start Frame** — First frame of the sequence; temporal state resets here.
- **End Frame** — Last frame for temporal denoising. Frames outside this range fall back to non-temporal (HDR) mode.

**Memory Settings**
- **Tile Size** — Controls how the image is broken into tiles for GPU processing:
  - *No Tiling* — Processes the entire image at once (uses more VRAM, fastest when VRAM allows)
  - *512×512* — Recommended for GPUs with ~8 GB VRAM
  - *1024×1024* — Recommended for GPUs with 12+ GB VRAM
  - *2048×2048* — Recommended for GPUs with 24+ GB VRAM

### Basic Workflow

```
Read (noisy CG render)
│
├── beauty ──► OptiXDenoiser ──► Write / Viewer
├── albedo ──┘ (optional)
├── normal ──┘ (optional, requires albedo)
└── motion ──┘ (optional, enables temporal mode)
```

A sample Nuke script (`OptixDenoiser.nk`) and test EXR (`DenoiseTEST001_.exr`) are included in the repo for quick testing.

---

## Building from Source

### Dependencies

- **CMake** 3.18+
- **CUDA Toolkit** 12.x (tested with 12.8)
- **NVIDIA OptiX SDK** 9.0
- **Nuke NDK** (ships with your Nuke installation)
- A C++17-compatible compiler (GCC on Linux, Visual Studio 2022 on Windows)

### Linux

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

cd LINUX
rm -rf build && mkdir build && cd build
cmake .. -DNUKE_VERSION=16.0v6
make -j$(nproc)
```

The resulting `OptiXDenoiser.so` will be in `build/lib/`. Verify linkage with `ldd ./OptiXDenoiser.so` — the CUDA runtime is statically linked, so you should **not** see `libcudart.so` in the output.

### Windows

```cmd
cd WIN
rmdir /s /q build
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DNUKE_VERSION=16.0v6 -DCMAKE_INSTALL_PREFIX=%USERPROFILE%\.nuke
cmake --build . --config Release
cmake --install . --config Release
```

### CMake Options

| Option | Default | Description |
|:-------|:--------|:------------|
| `NUKE_VERSION` | `16.0v6` | Target Nuke version (must match your installation) |
| `ENABLE_OPTIX_DEBUG` | `OFF` | Enable verbose OptiX debug logging |
| `ENABLE_ASAN` | `OFF` | Enable AddressSanitizer (Linux debug builds) |
| `ENABLE_UBSAN` | `OFF` | Enable UndefinedBehaviorSanitizer (Linux debug builds) |

> **Tip:** The OptiX SDK path defaults to `/opt/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64` on Linux. Update `OPTIX_ROOT_DIR` in `CMakeLists.txt` if your installation differs.

---

## Technical Details

- Built with **C++17** and the Nuke NDK `PlanarIop` API for full-plane GPU processing.
- Uses **CUDA streams** for asynchronous GPU operations.
- CUDA memory allocations are **128-byte aligned** and tracked for safe cleanup.
- The plugin uses `OPTIX_DENOISER_MODEL_KIND_HDR` for single-frame denoising and `OPTIX_DENOISER_MODEL_KIND_TEMPORAL` when motion vectors are connected.
- HDR input validation warns about pixel values exceeding 10,000 that may affect denoising quality.
- Normal buffer values are clamped to [0, 1] before being sent to the denoiser.
- Thread-safe resource management with mutex-protected allocation tracking and reverse-order deallocation.

---

## Project Structure

```
OptiXDenoiser/
├── src/
│   ├── OptiXDenoiser.cpp       # Plugin implementation
│   └── OptiXDenoiser.h         # Header with class definitions
├── LINUX/
│   ├── CMakeLists.txt           # Linux build configuration
│   ├── COMPILE_STEPBYSTEP.txt   # Quick build reference
│   └── <version>/               # Pre-compiled .so per Nuke version
├── WIN/
│   ├── CMakeLists.txt           # Windows build configuration
│   ├── COMPILE_STEPBYSTEP.txt   # Quick build reference
│   └── <version>/               # Pre-compiled .dll per Nuke version
├── OptixDenoiser.nk             # Sample Nuke script
├── DenoiseTEST001_.exr          # Test image
└── LICENSE                      # MIT License
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Author

**Peter Mercell** — [petermercell.com](https://petermercell.com)

---

## Acknowledgments

- [NVIDIA OptiX SDK](https://developer.nvidia.com/optix) for the AI-accelerated denoising technology.
- [Foundry Nuke NDK](https://learn.foundry.com/nuke/developers/) for the plugin development framework.
