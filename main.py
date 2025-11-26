#!/usr/bin/env python3
"""
Computer Vision Analysis Toolkit
Main entry point for analyzing any type of video
"""

import sys
import os
from config import INPUT_VIDEO, MAX_FRAMES
from preprocessing import preprocess_video, load_raw_video 

# Import flow modules
from modules.flow.lucas_kanade import analyze_lucas_kanade
from modules.flow.dis import analyze_dis
from modules.flow.velocity_magnitude import analyze_velocity_magnitude

# Import visual modules
from modules.visual.light_reflection import analyze_light_reflection
from modules.visual.light_rays import analyze_light_rays
from modules.visual.texture_analysis import analyze_texture

# Import pose modules
from modules.pose.human_pose import analyze_human_pose
from modules.pose.hand_tracking import analyze_hand_tracking
from modules.pose.face_mesh import analyze_face_mesh
from modules.pose.feature_tracking import analyze_feature_tracking


def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("  COMPUTER VISION ANALYSIS TOOLKIT")
    print("  Modular framework for video analysis")
    print("=" * 70)
    print()
def print_menu():
    """Print analysis options menu"""
    print("\n" + "=" * 70)
    print("AVAILABLE ANALYSIS METHODS")
    print("=" * 70)
    
    print("\n📊 FLOW ANALYSIS (Motion & Velocity)")
    print("  1. Lucas-Kanade Optical Flow (sparse feature tracking)")
    print("  2. DIS Optical Flow (dense inverse search)")
    print("  3. Velocity Magnitude (speed heatmap)")
    
    print("\n🎨 VISUAL ANALYSIS (Light & Texture)")
    print("  4. Light Reflection Analysis (sparkles & brightness)")
    print("  5. Light Ray Projection (UV/sun ray radiation)")
    print("  6. Surface Texture Analysis (turbulence & patterns)")
    
    print("\n🧍 POSE ANALYSIS (Human & Body Tracking)")
    print("  7. Human Pose Estimation (full body tracking)")
    print("  8. Hand Tracking (hand landmarks & gestures)")
    print("  9. Face Mesh (facial landmarks)")
    print("  10. Feature Tracking - ORB/AKAZE (anime-friendly)")  # NEW
    
    print("\n⚡ BATCH OPERATIONS")
    print("  11. Run ALL Flow analyses")      # Updated numbers
    print("  12. Run ALL Visual analyses")    # Updated numbers
    print("  13. Run ALL Pose analyses")      # Updated numbers
    print("  14. Run EVERYTHING")             # Updated numbers
    
    print("\n  0. Exit")
    print("=" * 70)
    print()

def get_analysis_categories():
    """Return organized analysis methods"""
    return {
        'flow': {
            '1': ('Lucas-Kanade Flow', analyze_lucas_kanade),
            '2': ('DIS Flow', analyze_dis),
            '3': ('Velocity Magnitude', analyze_velocity_magnitude)
        },
        'visual': {
            '4': ('Light Reflection', analyze_light_reflection),
            '5': ('Light Ray Projection', analyze_light_rays),
            '6': ('Texture Analysis', analyze_texture)
        },
        'pose': {
            '7': ('Human Pose Estimation', analyze_human_pose),
            '8': ('Hand Tracking', analyze_hand_tracking),
            '9': ('Face Mesh', analyze_face_mesh),
            '10': ('Feature Tracking (ORB/AKAZE)', analyze_feature_tracking)  # NEW
        }
    }

def run_analysis(choice, frames, metadata):
    """Run selected analysis"""
    
    categories = get_analysis_categories()
    
    # Flatten all analyses
    all_analyses = {}
    for category in categories.values():
        all_analyses.update(category)
    
    # Single analysis
    if choice in all_analyses:
        name, func = all_analyses[choice]
        print(f"\nStarting {name}...")
        output = func(frames, metadata)
        print(f"\n✓ {name} complete!")
        return [output]
    
    # Batch operations
    elif choice == '10':  # All Flow
        print("\nRunning ALL Flow analyses...")
        outputs = []
        for name, func in categories['flow'].values():
            print(f"\nStarting {name}...")
            output = func(frames, metadata)
            outputs.append(output)
            print(f"✓ {name} complete!")
        return outputs
    # Batch operations
    elif choice == '11':  # All Flow (was '10')
        print("\nRunning ALL Flow analyses...")
        outputs = []
        for name, func in categories['flow'].values():
            print(f"\nStarting {name}...")
            output = func(frames, metadata)
            outputs.append(output)
            print(f"✓ {name} complete!")
        return outputs
    
    elif choice == '12':  # All Visual (was '11')
        print("\nRunning ALL Visual analyses...")
        outputs = []
        for name, func in categories['visual'].values():
            print(f"\nStarting {name}...")
            output = func(frames, metadata)
            outputs.append(output)
            print(f"✓ {name} complete!")
        return outputs
    
    elif choice == '13':  # All Pose (was '12')
        print("\nRunning ALL Pose analyses...")
        outputs = []
        for name, func in categories['pose'].values():
            print(f"\nStarting {name}...")
            output = func(frames, metadata)
            outputs.append(output)
            print(f"✓ {name} complete!")
        return outputs
    
    elif choice == '14':  # Everything (was '13')
        print("\nRunning EVERYTHING...")
        outputs = []
        for category in categories.values():
            for name, func in category.values():
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
    else:
        print("\nLoading raw video (no preprocessing)...")
        frames, metadata = load_raw_video(INPUT_VIDEO, MAX_FRAMES)
        print("\n✓ Raw video loaded!")
    
    # Main menu loop
    while True:
        print_menu()
        choice = input("Select analysis method (0-13): ").strip()
        
        if choice == '0':
            print("\nExiting. Thank you for using the CV Analysis Toolkit!")
            sys.exit(0)
        
        valid_choices = [str(i) for i in range(15)]  # Was range(14), now range(15)
        if choice not in valid_choices:
            print("\n⚠ Invalid choice. Please select 0-13.")
            continue
        
        # Load/preprocess if not already done
        if frames is None:
            print("\nPreprocessing video...")
            frames, metadata = preprocess_video(INPUT_VIDEO, MAX_FRAMES)
            print("\n✓ Preprocessing complete!")
        
        # Run selected analysis
        outputs = run_analysis(choice, frames, metadata)
        
        if outputs:
            print("\n" + "=" * 70)
            print("ANALYSIS COMPLETE!")
            print("=" * 70)
            print("\nOutput files:")
            for output in outputs:
                print(f"  • {output}")
            print()
        
        # Ask if user wants to continue
        continue_choice = input("Run another analysis? [Y/n]: ").strip().lower()
        if continue_choice in ['n', 'no']:
            print("\nThank you for using the CV Analysis Toolkit!")
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