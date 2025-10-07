#!/usr/bin/env python3
"""
Example script showing how to use the OMR system without UI
"""

import main

# Example of how to run the OMR system programmatically
def run_omr_example():
    # Clear any previous results
    main.id.clear()
    main.res.clear()
    
    # Example image paths (replace with your actual image paths)
    image_paths = [
        "omr.png",  # Default OMR image in the project
        # Add more image paths here as needed
    ]
    
    print("=== OMR Processing Example ===")
    
    for image_path in image_paths:
        try:
            print(f"Processing: {image_path}")
            main.processImages(image_path)
            print(f"✓ Successfully processed {image_path}")
        except Exception as e:
            print(f"✗ Error processing {image_path}: {str(e)}")
    
    # Save results
    if main.id and main.res:
        main.save_to_excel(main.id, main.res)
        print("\nResults:")
        for student_id, score in zip(main.id, main.res):
            print(f"Student {student_id}: {score:.1f}%")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_omr_example()