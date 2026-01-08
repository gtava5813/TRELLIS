"""
Example: Style Consistency for TRELLIS 3D Generation
=====================================================

This example demonstrates how to generate multiple 3D assets with
consistent artistic style using the same style parameters.

Use cases:
- Generate a set of matching furniture (chair, table, lamp with same style)
- Create buildings of different sizes with consistent architectural style
- Produce a collection of game assets with unified visual theme

Requirements:
- TRELLIS installed with all dependencies
- GPU with at least 16GB memory
"""

import os
os.environ['SPCONV_ALGO'] = 'native'
# os.environ['ATTN_BACKEND'] = 'xformers'  # Uncomment if flash-attn unavailable

import json
import imageio
import numpy as np
from pathlib import Path
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


def main():
    # Create output directory
    output_dir = Path("output_style_consistency")
    output_dir.mkdir(exist_ok=True)

    # Load the text-to-3D model
    print("Loading TRELLIS-text-xlarge model...")
    print("(This may take a few minutes on first run)")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
    pipeline.cuda()
    print("✓ Model loaded!\n")

    # =========================================================================
    # Example 1: Generate multiple buildings with same style
    # =========================================================================
    print("=" * 60)
    print("Example 1: Same Style, Different Sizes")
    print("=" * 60)

    # Define a theme prefix for consistent style description
    theme_prefix = "low poly cyberpunk, dark moody atmosphere, neon lights, "
    
    # Define the style parameters
    style = TrellisTextTo3DPipeline.extract_style(
        seed=1234,  # Fixed seed for reproducibility
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,  # Higher = more adherence to prompt
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 3.0,
        },
    )

    # Generate buildings of different sizes with same style
    buildings = [
        ("large_building", "a large skyscraper building"),
        ("medium_building", "a medium office building"),
        ("small_building", "a small shop building"),
    ]

    for name, prompt in buildings:
        full_prompt = theme_prefix + prompt
        print(f"\nGenerating: {name}")
        print(f"  Prompt: {full_prompt}")
        
        outputs, _ = pipeline.run_with_style(
            full_prompt,
            style_params=style,
        )
        
        # Export GLB
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.9,
            texture_size=1024,
        )
        glb_path = output_dir / f"{name}.glb"
        glb.export(str(glb_path))
        print(f"  ✓ Saved: {glb_path}")
        
        # Save preview video
        video = render_utils.render_video(outputs['mesh'][0])['normal']
        video_path = output_dir / f"{name}_preview.mp4"
        imageio.mimsave(str(video_path), video, fps=30)

    # =========================================================================
    # Example 2: Generate and reuse style from first generation
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 2: Extract Style from First Generation")
    print("=" * 60)

    # Generate first asset and extract its style
    print("\nGenerating chair and extracting style...")
    outputs, extracted_style = pipeline.run_with_style(
        "a futuristic gaming chair with RGB lighting",
        seed=42,
    )
    
    # Save the chair
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.9,
        texture_size=1024,
    )
    glb.export(str(output_dir / "chair.glb"))
    print("  ✓ Saved: chair.glb")
    
    # Save style to JSON for later use
    style_file = output_dir / "extracted_style.json"
    with open(style_file, 'w') as f:
        json.dump(extracted_style, f, indent=2)
    print(f"  ✓ Style saved to: {style_file}")
    
    # Apply same style to matching furniture
    matching_items = [
        ("desk", "a futuristic gaming desk"),
        ("monitor", "a curved gaming monitor"),
        ("mousepad", "a large RGB mousepad"),
    ]
    
    print("\nGenerating matching items with same style...")
    for name, prompt in matching_items:
        outputs, _ = pipeline.run_with_style(
            prompt,
            style_params=extracted_style,
        )
        
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.9,
            texture_size=1024,
        )
        glb.export(str(output_dir / f"{name}.glb"))
        print(f"  ✓ Saved: {name}.glb")

    # =========================================================================
    # Example 3: Style Interpolation
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 3: Style Interpolation")
    print("=" * 60)

    # Define two contrasting styles
    style_a = TrellisTextTo3DPipeline.extract_style(
        seed=100,
        sparse_structure_sampler_params={"steps": 8, "cfg_strength": 5.0},
        slat_sampler_params={"steps": 8, "cfg_strength": 2.0},
    )
    
    style_b = TrellisTextTo3DPipeline.extract_style(
        seed=100,  # Same seed for structure
        sparse_structure_sampler_params={"steps": 16, "cfg_strength": 10.0},
        slat_sampler_params={"steps": 16, "cfg_strength": 5.0},
    )
    
    # Interpolate between styles
    print("\nGenerating with interpolated styles (0%, 50%, 100%)...")
    for alpha in [0.0, 0.5, 1.0]:
        interpolated_style = TrellisTextTo3DPipeline.interpolate_styles(style_a, style_b, alpha)
        
        outputs, _ = pipeline.run_with_style(
            "a modern house",
            style_params=interpolated_style,
        )
        
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.9,
            texture_size=1024,
        )
        glb.export(str(output_dir / f"house_style_{int(alpha*100)}.glb"))
        print(f"  ✓ Saved: house_style_{int(alpha*100)}.glb (alpha={alpha})")

    print("\n" + "=" * 60)
    print(f"✓ All outputs saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
