"""
TRELLIS Colab Setup Script
==========================
A Python-based setup script that works directly in Google Colab cells.

Usage in Colab:
    !git clone --recurse-submodules https://github.com/gtava5813/TRELLIS.git
    %cd TRELLIS
    from setup_colab import setup_trellis
    setup_trellis()
"""


import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, desc=None, check=True):
    """Run a shell command with optional description."""
    if desc:
        print(f"📦 {desc}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=check, 
            capture_output=True, text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def get_cuda_version():
    """Detect CUDA version from PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"✅ CUDA {cuda_version} detected")
            return cuda_version
        else:
            print("⚠️  CUDA not available, some features will be disabled")
            return None
    except ImportError:
        return None


def get_pytorch_version():
    """Get PyTorch version."""
    try:
        import torch
        version = torch.__version__.split('+')[0]
        print(f"✅ PyTorch {version} detected")
        return version
    except ImportError:
        return None


def install_basic_dependencies():
    """Install basic pip dependencies from requirements.txt."""
    print("\n" + "="*50)
    print("📦 STEP 1: Installing Basic Dependencies")
    print("="*50 + "\n")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        run_command(
            f"{sys.executable} -m pip install -r {requirements_file} -q",
            "Installing from requirements.txt"
        )
    else:
        # Fallback: install individually
        packages = [
            "pillow", "imageio", "imageio-ffmpeg", "tqdm", "easydict",
            "opencv-python-headless", "scipy", "ninja", "rembg", "onnxruntime",
            "trimesh", "open3d", "xatlas", "pyvista", "pymeshfix", "igraph",
            "transformers", "huggingface-hub", "safetensors"
        ]
        run_command(
            f"{sys.executable} -m pip install {' '.join(packages)} -q",
            "Installing basic packages"
        )
    
    # Install utils3d from git
    run_command(
        f"{sys.executable} -m pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 -q",
        "Installing utils3d"
    )
    print("✅ Basic dependencies installed\n")


def install_xformers(pytorch_version, cuda_version):
    """Install xformers for the detected PyTorch/CUDA version."""
    print("\n" + "="*50)
    print("📦 STEP 2: Installing xformers")
    print("="*50 + "\n")
    
    if cuda_version is None:
        print("⚠️  Skipping xformers (no CUDA)")
        return
    
    cuda_major = cuda_version.split('.')[0]
    
    # Map PyTorch versions to xformers versions
    xformers_map = {
        ("2.4.0", "11"): "xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118",
        ("2.4.0", "12"): "xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121",
        ("2.4.1", "11"): "xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu118",
        ("2.4.1", "12"): "xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121",
        ("2.5.0", "11"): "xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu118",
        ("2.5.0", "12"): "xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu121",
    }
    
    key = (pytorch_version, cuda_major)
    if key in xformers_map:
        run_command(
            f"{sys.executable} -m pip install {xformers_map[key]} -q",
            f"Installing xformers for PyTorch {pytorch_version} + CUDA {cuda_major}"
        )
        print("✅ xformers installed\n")
    else:
        print(f"⚠️  No pre-built xformers for PyTorch {pytorch_version} + CUDA {cuda_major}")
        print("   Trying generic install...")
        run_command(f"{sys.executable} -m pip install xformers -q", check=False)


def install_flash_attn():
    """Install flash-attention."""
    print("\n" + "="*50)
    print("📦 STEP 3: Installing flash-attn")
    print("="*50 + "\n")
    
    success, _ = run_command(
        f"{sys.executable} -m pip install flash-attn --no-build-isolation -q",
        "Installing flash-attn (this may take a few minutes)",
        check=False
    )
    
    if success:
        print("✅ flash-attn installed\n")
    else:
        print("⚠️  flash-attn installation failed")
        print("   Will use xformers as fallback")
        print("   Set: os.environ['ATTN_BACKEND'] = 'xformers'\n")


def install_spconv(cuda_version):
    """Install spconv for the detected CUDA version."""
    print("\n" + "="*50)
    print("📦 STEP 4: Installing spconv")
    print("="*50 + "\n")
    
    if cuda_version is None:
        print("⚠️  Skipping spconv (no CUDA)")
        return
    
    cuda_major = cuda_version.split('.')[0]
    cuda_minor = cuda_version.split('.')[1] if '.' in cuda_version else "0"
    
    # spconv only has cu118 and cu120 builds, not cu121
    # For CUDA 12.x, use spconv-cu120 which is compatible
    if cuda_major == "11":
        package = "spconv-cu118"
    elif cuda_major == "12":
        package = "spconv-cu120"  # Works for 12.1, 12.2, etc.
    else:
        print(f"⚠️  Unknown CUDA version {cuda_version}, trying spconv-cu120")
        package = "spconv-cu120"
    
    success, _ = run_command(
        f"{sys.executable} -m pip install {package} -q",
        f"Installing {package}",
        check=False
    )
    
    if success:
        print("✅ spconv installed\n")
    else:
        print(f"⚠️  {package} installation failed")
        print("   This may be due to Python version incompatibility")
        print("   Trying alternative installation method...")
        # Try building from source as fallback
        run_command(
            f"{sys.executable} -m pip install spconv -q",
            "Trying generic spconv",
            check=False
        )


def install_cuda_extensions():
    """Install CUDA extensions (diffoctreerast, nvdiffrast, mip-splatting)."""
    print("\n" + "="*50)
    print("📦 STEP 5: Installing CUDA Extensions")
    print("="*50)
    print("⚠️  This step requires compilation and may take 10-15 minutes\n")
    
    import tempfile
    ext_dir = Path(tempfile.gettempdir()) / "trellis_extensions"
    ext_dir.mkdir(exist_ok=True)
    
    # nvdiffrast
    print("📦 Installing nvdiffrast...")
    nvdiffrast_dir = ext_dir / "nvdiffrast"
    if not nvdiffrast_dir.exists():
        run_command(f"git clone https://github.com/NVlabs/nvdiffrast.git {nvdiffrast_dir}", check=False)
    run_command(f"{sys.executable} -m pip install {nvdiffrast_dir} -q", check=False)
    
    # diffoctreerast
    print("📦 Installing diffoctreerast...")
    diffoctreerast_dir = ext_dir / "diffoctreerast"
    if not diffoctreerast_dir.exists():
        run_command(f"git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git {diffoctreerast_dir}", check=False)
    run_command(f"{sys.executable} -m pip install {diffoctreerast_dir} -q", check=False)
    
    # mip-splatting (gaussian rasterization)
    print("📦 Installing mip-splatting gaussian rasterization...")
    mipsplatting_dir = ext_dir / "mip-splatting"
    if not mipsplatting_dir.exists():
        run_command(f"git clone https://github.com/autonomousvision/mip-splatting.git {mipsplatting_dir}", check=False)
    diff_gaussian_dir = mipsplatting_dir / "submodules" / "diff-gaussian-rasterization"
    if diff_gaussian_dir.exists():
        run_command(f"{sys.executable} -m pip install {diff_gaussian_dir} -q", check=False)
    
    # kaolin
    print("📦 Installing kaolin...")
    try:
        import torch
        pytorch_version = torch.__version__.split('+')[0]
        cuda_version = torch.version.cuda
        
        kaolin_urls = {
            "2.4.0": "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html",
            "2.2.2": "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.2_cu118.html",
        }
        
        if pytorch_version in kaolin_urls:
            run_command(
                f"{sys.executable} -m pip install kaolin -f {kaolin_urls[pytorch_version]} -q",
                check=False
            )
    except Exception as e:
        print(f"⚠️  kaolin installation failed: {e}")
    
    print("✅ CUDA extensions installation attempted\n")


def install_vox2seq():
    """Install vox2seq extension from local extensions folder."""
    print("\n📦 Installing vox2seq...")
    
    vox2seq_dir = Path(__file__).parent / "extensions" / "vox2seq"
    if vox2seq_dir.exists():
        import tempfile
        import shutil
        
        temp_dir = Path(tempfile.gettempdir()) / "vox2seq"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        shutil.copytree(vox2seq_dir, temp_dir)
        
        run_command(f"{sys.executable} -m pip install {temp_dir} -q", check=False)
        print("✅ vox2seq installed\n")
    else:
        print("⚠️  vox2seq extension not found in extensions/\n")


def verify_installation():
    """Verify all components are properly installed."""
    print("\n" + "="*50)
    print("🔍 VERIFICATION")
    print("="*50 + "\n")
    
    components = {
        "PyTorch": "import torch; print(f'  Version: {torch.__version__}')",
        "CUDA": "import torch; print(f'  Available: {torch.cuda.is_available()}')",
        "xformers": "import xformers; print('  ✅ Installed')",
        "flash_attn": "import flash_attn; print('  ✅ Installed')",
        "spconv": "import spconv; print('  ✅ Installed')",
        "transformers": "import transformers; print('  ✅ Installed')",
        "open3d": "import open3d; print('  ✅ Installed')",
        "rembg": "import rembg; print('  ✅ Installed')",
        "TRELLIS Pipeline": "from trellis.pipelines import TrellisImageTo3DPipeline; print('  ✅ Installed')",
    }
    
    success_count = 0
    for name, check_code in components.items():
        print(f"{name}:")
        try:
            exec(check_code)
            success_count += 1
        except Exception as e:
            print(f"  ❌ Not available: {str(e)[:50]}")
    
    print(f"\n{'='*50}")
    print(f"✅ {success_count}/{len(components)} components verified")
    print("="*50 + "\n")
    
    return success_count >= 6  # At least core components


def setup_trellis(
    install_flash_attention=True,
    install_extensions=True,
    verify=True
):
    """
    One-liner setup for TRELLIS on Google Colab.
    
    Args:
        install_flash_attention: Whether to install flash-attn (can be slow)
        install_extensions: Whether to install CUDA extensions (slow, ~10-15 min)
        verify: Whether to run verification after installation
    
    Example:
        from setup_colab import setup_trellis
        setup_trellis()
    """
    print("="*50)
    print("🚀 TRELLIS Setup for Google Colab")
    print("="*50)
    
    # Detect environment
    pytorch_version = get_pytorch_version()
    cuda_version = get_cuda_version()
    
    if pytorch_version is None:
        print("❌ PyTorch not found. Please install PyTorch first.")
        print("   Run: !pip install torch torchvision")
        return False
    
    # Install dependencies
    install_basic_dependencies()
    install_xformers(pytorch_version, cuda_version)
    
    if install_flash_attention and cuda_version:
        install_flash_attn()
    else:
        print("\n⚠️  Skipping flash-attn, using xformers backend")
        os.environ['ATTN_BACKEND'] = 'xformers'
    
    if cuda_version:
        install_spconv(cuda_version)
    
    if install_extensions and cuda_version:
        install_cuda_extensions()
        install_vox2seq()
    else:
        print("\n⚠️  Skipping CUDA extensions")
    
    # Verify
    if verify:
        success = verify_installation()
        if success:
            print("🎉 TRELLIS setup complete! You're ready to generate 3D assets.")
        else:
            print("⚠️  Some components failed. Check the output above.")
        return success
    
    return True


def quick_setup():
    """Quick setup without CUDA extensions (faster, basic functionality)."""
    return setup_trellis(
        install_flash_attention=False,
        install_extensions=False,
        verify=True
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TRELLIS Colab Setup")
    parser.add_argument("--quick", action="store_true", help="Quick setup without CUDA extensions")
    parser.add_argument("--no-flash-attn", action="store_true", help="Skip flash-attention")
    parser.add_argument("--no-extensions", action="store_true", help="Skip CUDA extensions")
    args = parser.parse_args()
    
    if args.quick:
        quick_setup()
    else:
        setup_trellis(
            install_flash_attention=not args.no_flash_attn,
            install_extensions=not args.no_extensions
        )
