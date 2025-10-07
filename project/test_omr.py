#!/usr/bin/env python3
"""
Test script for the modified OMR system without UI
"""

import main

def test_single_image():
    """Test processing a single image"""
    # Clear any previous results
    main.id.clear()
    main.res.clear()
    
    print("=== OMR Test - Single Image ===")
    
    # Test with the default OMR image
    image_path = "omr.png"
    
    try:
        print(f"Testing with: {image_path}")
        main.processImages(image_path)
        
        if main.id and main.res:
            # main.save_to_excel(main.id, main.res)  # No longer saving to Excel
            print("\nTest completed successfully! Results displayed above.")
        else:
            print("No results generated.")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_single_image()