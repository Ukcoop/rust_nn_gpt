{ pkgs ? import <nixpkgs> {} }:

let
  # Use nixpkgs with your system's settings
  nixos = import <nixpkgs> {};
in
pkgs.mkShell {
  buildInputs = [
    # Rust toolchain
    nixos.pkg-config  # Required for some Rust dependencies
    nixos.cmake       # Required for some Rust dependencies
    nixos.openssl     # Common dependency
    nixos.openssl.dev

    # Required for WebGPU (wgpu)
    nixos.vulkan-headers
    nixos.vulkan-loader
    nixos.vulkan-tools
    nixos.glslang      # For glslangValidator (GLSL to SPIR-V)

    # GPU-specific drivers
    nixos.mesa               # Open-source drivers (AMD/Intel)
    nixos.libGL              # For OpenGL support

    # Intel GPU compute support
    nixos.intel-compute-runtime  # Intel NEO OpenCL runtime
    nixos.intel-media-driver     # Intel media driver for VAAPI
    nixos.intel-vaapi-driver     # Additional Intel VAAPI driver
    nixos.ocl-icd                # OpenCL ICD Loader
    nixos.clinfo                 # For OpenCL diagnostics

    # X11 dependencies required by wgpu
    nixos.xorg.libX11
    nixos.xorg.libXcursor
    nixos.xorg.libXrandr
    nixos.xorg.libXi

    # Optional: for GPU debugging
    nixos.vulkan-validation-layers
    nixos.glxinfo              # For OpenGL diagnostics
  ];

  # Set environment variables that may be useful
  shellHook = ''
    # Rust-specific configuration
    export RUST_BACKTRACE=1
    export RUSTFLAGS="-C target-cpu=native"
    
    # Cargo configuration
    export CARGO_HOME="$HOME/.cargo"
    export PATH="$CARGO_HOME/bin:$PATH"
    
    # Vulkan configuration
    export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
    export VK_LAYER_PATH=${nixos.vulkan-validation-layers}/share/vulkan/explicit_layer.d
    export VK_ICD_FILENAMES=${nixos.mesa}/share/vulkan/icd.d/intel_icd.x86_64.json
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${nixos.vulkan-loader}/lib:${nixos.mesa}/lib
    
    # GPU selection (0 = Intel, 1 = discrete GPU if available)
    export DRI_PRIME=''${DRI_PRIME:-0}
    
    # Wayland/X11 GPU device access
    export XDG_RUNTIME_DIR=''${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
    
    # Intel GPU and OpenCL configuration
    export OPENCL_VENDOR_PATH="$HOME/.config/OpenCL/vendors"
    export OCL_ICD_FILENAMES="$OPENCL_VENDOR_PATH/intel.icd"
    export INTEL_OPENCL_ICD=${nixos.intel-compute-runtime}/lib/intel-opencl/libigdrcl.so
    export LIBVA_DRIVER_NAME=iHD
    export LIBVA_DRIVERS_PATH=${nixos.intel-vaapi-driver}/lib/dri
    
    # WebGPU configuration
    export WGPU_POWER_PREF=high-performance
    export WGPU_DX12_COMPILER=dxc
    
    # Create user-specific ICD file for Intel GPU compute
    mkdir -p $HOME/.config/OpenCL/vendors
    echo "${nixos.intel-compute-runtime}/lib/intel-opencl/libigdrcl.so" > $HOME/.config/OpenCL/vendors/intel.icd
    echo "Environment ready! You can now use 'cargo run' directly."
  '';
}
