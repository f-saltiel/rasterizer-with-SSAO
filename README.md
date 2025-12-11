# Screen Space Ambient Occlusion in a CPU Rasterizer

This repository contains the final project for the Computer Graphics course. It implements Screen Space Ambient Occlusion (SSAO) inside a custom CPU-based rasterizer written in C++. The renderer outputs 1200×1200 PPM images and performs all processing on the CPU with no GPU acceleration.

## Features
- Custom CPU rasterizer (scanline triangle rasterization, z-buffering, Phong shading)
- Per-pixel G-buffer data (depth, view-space position, normal, shaded color)
- SSAO implementation using:
  - 32-sample hemisphere kernel
  - 4×4 noise texture for per-pixel kernel rotation
  - View-space sampling and depth comparison
  - Bilateral blur (spatial + depth weighting)
  - Final modulation of ambient lighting

## Scene Contents
The test scene includes:
- Three skyscraper OBJ models
- A subdivided procedural floor
- A Mini Cooper model near the camera
- A Cessna airplane model above the ground

Models sourced from:
https://people.sc.fsu.edu/~jburkardt/data/obj/obj.html

## Repository Structure
```
.
├── Rasterizer.cpp          # Full rasterizer + SSAO implementation
├── rasterOutput.ppm        # Render with no SSAO applied
├── final_ssao_render.ppm   # Render with SSAO applied
├── models/                 # OBJ models used in the scene
│   ├── skyscraper.obj
│   ├── minicooper.obj
│   └── cessna.obj
├── debug/                  # Debugging maps
│   ├── debug_ssao_raw.ppm
│   ├── debug_ssao_blur.ppm
│   ├── debug_normals.ppm
│   ├── debug_depth.ppm
│   └── debug_positions.ppm
└── README.md
```

## Build Instructions

Requires a C++17 compiler.

Compile:
```
g++ -std=c++17 -O2 Rasterizer.cpp -o rasterizer
```

Run:
```
./rasterizer
```
