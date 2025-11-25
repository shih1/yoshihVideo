#!/usr/bin/env python3
"""
Water Video Analysis Framework
Main entry point for analyzing water stream videos
"""

import sys
import os
from config import INPUT_VIDEO, MAX_FRAMES
from preprocessing import preprocess_video
from flow_lucas_kanade import analyze_lucas_kanade
from flow_dis import analyze_dis
from light_reflection import analyze_light_reflection
from texture_analysis import analyze_texture
from velocity_magnitude import analyze_velocity_magnitude
from light_rays import analyze_light_rays

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("  WATER VIDEO ANALYSIS FRAMEWORK")
    print("=" * 60)
    print()

def print_menu():
    """Print analysis options menu"""
    print("\nAvailable Analysis Methods:")
    print("  1. Lucas-Kanade Optical Flow (sparse feature tracking)")
    print("  2. DIS Optical Flow (dense inverse search)")
    print("  3. Light Reflection Analysis (sparkles & brightness)")
    print("  4. Surface Texture Analysis (turbulence & patterns)")
    print("  5. Velocity Magnitude (speed heatmap - highlights waterfalls)")
    print("  6. Light Ray Projection (UV/sun ray radiation)")
    print("  7. Run ALL analyses")
    print("  0. Exit")
    print()

def run_analysis(choice, frames, metadata):
    """Run selected analysis"""
    
    analyses = {
        '1': ('Lucas-Kanade Flow', analyze_lucas_kanade),
        '2': ('DIS Flow', analyze_dis),
        '3': ('Light Reflection', analyze_light_reflection),
        '4': ('Texture Analysis', analyze_texture),
        '5': ('Velocity Magnitude', analyze_velocity_magnitude),
        '6': ('Light Ray Projection', analyze_light_rays)
    }
    
    if choice in analyses:
        name, func = analyses[choice]
        print(f"\nStarting {name}...")
        output = func(frames, metadata)
        print(f"\n✓ {name} complete!")
        return [output]
    
    elif choice == '7':
        print("\nRunning ALL analyses...")
        outputs = []
        for name, func in analyses.values():
            print(f"\nStarting {name}...")
            output = func(frames, metadata)
            outputs.append(output)
            print(f"✓ {name} complete!")
        return outputs
    
    return []

def main():
    """Main program loop"""
    print_banner()
    
    # Check if video file exists
    if not os.path.exists(INPUT_VIDEO):
        print(f"ERROR: Video file '{INPUT_VIDEO}' not found!")
        print(f"Please place your video file in the current directory or update config.py")
        sys.exit(1)
    
    print(f"Video file: {INPUT_VIDEO}")
    print(f"Max frames to process: {MAX_FRAMES if MAX_FRAMES else 'ALL'}")
    print()
    
    # Ask if user wants to preprocess now or on-demand
    print("Preprocessing includes:")
    print("  - Video stabilization (removes camera shake)")
    print("  - Color enhancement (CLAHE, saturation, warmth)")
    print()
    
    preprocess_choice = input("Preprocess video now? (recommended) [Y/n]: ").strip().lower()
    
    frames = None
    metadata = None
    
    if preprocess_choice in ['', 'y', 'yes']:
        print("\nPreprocessing video...")
        frames, metadata = preprocess_video(INPUT_VIDEO, MAX_FRAMES)
        print("\n✓ Preprocessing complete!")
    
    # Main menu loop
    while True:
        print_menu()
        choice = input("Select analysis method (0-7): ").strip()
        
        if choice == '0':
            print("\nExiting. Thank you for using the Water Analysis Framework!")
            sys.exit(0)
        
        if choice not in ['1', '2', '3', '4', '5', '6', '7']:
            print("\n⚠ Invalid choice. Please select 0-7.")
            continue
        
        # Load/preprocess if not already done
        if frames is None:
            print("\nPreprocessing video...")
            frames, metadata = preprocess_video(INPUT_VIDEO, MAX_FRAMES)
            print("\n✓ Preprocessing complete!")
        
        # Run selected analysis
        outputs = run_analysis(choice, frames, metadata)
        
        if outputs:
            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE!")
            print("=" * 60)
            print("\nOutput files:")
            for output in outputs:
                print(f"  • {output}")
            print()
        
        # Ask if user wants to continue
        continue_choice = input("Run another analysis? [Y/n]: ").strip().lower()
        if continue_choice in ['n', 'no']:
            print("\nThank you for using the Water Analysis Framework!")
            sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)