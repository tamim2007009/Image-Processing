import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

class RegionDescriptors:
    def __init__(self):
        self.descriptors = {}
    
    def create_sample_shapes(self):
        """Create sample binary images of different shapes for demonstration"""
        
        # Create a rectangle
        rectangle = np.zeros((100, 100), dtype=np.uint8)
        rectangle[20:70, 30:80] = 1
        
        # Create a circle (approximate)
        circle = np.zeros((100, 100), dtype=np.uint8)
        center_x, center_y = 50, 50
        radius = 25
        for i in range(100):
            for j in range(100):
                if ((i - center_x)**2 + (j - center_y)**2) <= radius**2:
                    circle[i, j] = 1
        
        # Create a triangle (approximate)
        triangle = np.zeros((100, 100), dtype=np.uint8)
        for i in range(20, 80):
            for j in range(30, 70):
                if j >= 30 + (i - 20) * 0.5 and j <= 70 - (i - 20) * 0.5:
                    triangle[i, j] = 1
        
        return {'rectangle': rectangle, 'circle': circle, 'triangle': triangle}
    
    def erosion_manual(self, binary_image, structuring_element):
        """
        Manual implementation of morphological erosion using loops
        Erosion shrinks the white regions in binary image
        """
        height, width = binary_image.shape
        se_height, se_width = structuring_element.shape
        
        # Create output image
        eroded = np.zeros_like(binary_image)
        
        # Calculate padding needed
        pad_h = se_height // 2
        pad_w = se_width // 2
        
        # Pad the input image
        padded = np.zeros((height + 2*pad_h, width + 2*pad_w))
        padded[pad_h:pad_h+height, pad_w:pad_w+width] = binary_image
        
        print("Performing erosion...")
        
        # Apply erosion using nested loops
        for i in range(height):
            for j in range(width):
                # Extract region under structuring element
                region = padded[i:i+se_height, j:j+se_width]
                
                # Check if structuring element fits completely
                # Erosion: output is 1 only if all pixels under SE are 1
                fit = True
                for si in range(se_height):
                    for sj in range(se_width):
                        if structuring_element[si, sj] == 1:  # Only check where SE is 1
                            if region[si, sj] == 0:
                                fit = False
                                break
                    if not fit:
                        break
                
                eroded[i, j] = 1 if fit else 0
        
        return eroded
    
    def find_boundary(self, binary_image):
        """
        Find boundary by subtracting eroded image from original
        Boundary = Original - Eroded
        """
        print("Finding boundary...")
        
        # Create a 3x3 structuring element (cross shape)
        structuring_element = np.array([[0, 1, 0],
                                       [1, 1, 1],
                                       [0, 1, 0]], dtype=np.uint8)
        
        # Apply erosion
        eroded = self.erosion_manual(binary_image, structuring_element)
        
        # Boundary = Original - Eroded
        boundary = np.zeros_like(binary_image)
        height, width = binary_image.shape
        
        for i in range(height):
            for j in range(width):
                if binary_image[i, j] == 1 and eroded[i, j] == 0:
                    boundary[i, j] = 1
        
        return boundary
    
    def calculate_area(self, binary_image):
        """Calculate area by counting non-zero pixels manually"""
        print("Calculating area...")
        area = 0
        height, width = binary_image.shape
        
        for i in range(height):
            for j in range(width):
                if binary_image[i, j] == 1:
                    area += 1
        
        return area
    
    def calculate_perimeter(self, boundary_image):
        """Calculate perimeter by counting boundary pixels manually"""
        print("Calculating perimeter...")
        perimeter = 0
        height, width = boundary_image.shape
        
        for i in range(height):
            for j in range(width):
                if boundary_image[i, j] == 1:
                    perimeter += 1
        
        return perimeter
    
    def calculate_max_diameter(self, binary_image):
        """
        Calculate maximum diameter as max(width, height) of bounding box
        MaxDiameter = max(x_max - x_min, y_max - y_min)
        """
        print("Calculating max diameter...")
        height, width = binary_image.shape
        
        # Find bounding box coordinates manually
        min_row, max_row = height, -1
        min_col, max_col = width, -1
        
        for i in range(height):
            for j in range(width):
                if binary_image[i, j] == 1:
                    if i < min_row:
                        min_row = i
                    if i > max_row:
                        max_row = i
                    if j < min_col:
                        min_col = j
                    if j > max_col:
                        max_col = j
        
        # Calculate dimensions
        if max_row == -1:  # No object found
            return 0
        
        height_diff = max_row - min_row + 1
        width_diff = max_col - min_col + 1
        max_diameter = max(height_diff, width_diff)
        
        print(f"Bounding box: ({min_row}, {min_col}) to ({max_row}, {max_col})")
        print(f"Width: {width_diff}, Height: {height_diff}")
        
        return max_diameter
    
    def calculate_descriptors(self, area, perimeter, max_diameter):
        """
        Calculate shape descriptors:
        - Form Factor = 4π × Area / Perimeter²
        - Roundness = 4 × Area / (π × Max_Diameter²)  
        - Compactness = Perimeter² / Area
        """
        print("Calculating descriptors...")
        
        if perimeter == 0 or max_diameter == 0 or area == 0:
            return 0, 0, 0
        
        pi = 3.14159265359
        
        # Form Factor: measures how close the shape is to a circle
        # Perfect circle = 1, other shapes < 1
        form_factor = (4 * pi * area) / (perimeter * perimeter)
        
        # Roundness: another circularity measure
        # Perfect circle = 1, other shapes < 1
        roundness = (4 * area) / (pi * max_diameter * max_diameter)
        
        # Compactness: ratio of perimeter² to area
        # Circle has minimum compactness, complex shapes have higher values
        compactness = (perimeter * perimeter) / area
        
        return form_factor, roundness, compactness
    
    def process_image(self, binary_image, image_name):
        """Process a single binary image and extract all descriptors"""
        print(f"\n=== Processing {image_name} ===")
        
        # Step 1: Find boundary
        boundary = self.find_boundary(binary_image)
        
        # Step 2: Calculate basic parameters
        area = self.calculate_area(binary_image)
        perimeter = self.calculate_perimeter(boundary)
        max_diameter = self.calculate_max_diameter(binary_image)
        
        print(f"Area: {area}")
        print(f"Perimeter: {perimeter}")
        print(f"Max Diameter: {max_diameter}")
        
        # Step 3: Calculate descriptors
        form_factor, roundness, compactness = self.calculate_descriptors(
            area, perimeter, max_diameter)
        
        print(f"Form Factor: {form_factor:.6f}")
        print(f"Roundness: {roundness:.6f}")
        print(f"Compactness: {compactness:.6f}")
        
        # Store results
        self.descriptors[image_name] = {
            'area': area,
            'perimeter': perimeter,
            'max_diameter': max_diameter,
            'form_factor': form_factor,
            'roundness': roundness,
            'compactness': compactness,
            'boundary': boundary
        }
        
        return form_factor, roundness, compactness
    
    def calculate_similarity(self, desc1, desc2):
        """
        Calculate similarity between two descriptor sets
        Using Euclidean distance in normalized feature space
        """
        # Extract descriptor values
        features1 = [desc1['form_factor'], desc1['roundness'], desc1['compactness']]
        features2 = [desc2['form_factor'], desc2['roundness'], desc2['compactness']]
        
        # Calculate Euclidean distance manually
        distance = 0
        for i in range(len(features1)):
            diff = features1[i] - features2[i]
            distance += diff * diff
        
        distance = distance ** 0.5  # Square root
        
        # Convert distance to similarity (0 to 1, where 1 is identical)
        similarity = 1 / (1 + distance)
        
        return similarity
    
    def match_descriptors(self, train_names, test_names):
        """Match test images with training images based on descriptors"""
        print("\n=== Feature Matching Results ===")
        
        # Create similarity matrix
        similarity_matrix = []
        
        for test_name in test_names:
            similarities = []
            test_desc = self.descriptors[test_name]
            
            print(f"\nMatching {test_name}:")
            for train_name in train_names:
                train_desc = self.descriptors[train_name]
                similarity = self.calculate_similarity(test_desc, train_desc)
                similarities.append(similarity)
                print(f"  vs {train_name}: {similarity:.6f}")
            
            similarity_matrix.append(similarities)
            
            # Find best match
            best_match_idx = 0
            best_similarity = similarities[0]
            for i in range(1, len(similarities)):
                if similarities[i] > best_similarity:
                    best_similarity = similarities[i]
                    best_match_idx = i
            
            print(f"  Best match: {train_names[best_match_idx]} (similarity: {best_similarity:.6f})")
        
        return similarity_matrix
    
    def save_descriptors(self, filename="descriptors.txt"):
        """Save descriptors to text file"""
        with open(filename, 'w') as f:
            f.write("Image Processing Lab 5 - Region Descriptors\n")
            f.write("=" * 50 + "\n\n")
            
            for name, desc in self.descriptors.items():
                f.write(f"Image: {name}\n")
                f.write(f"Area: {desc['area']}\n")
                f.write(f"Perimeter: {desc['perimeter']}\n")
                f.write(f"Max Diameter: {desc['max_diameter']}\n")
                f.write(f"Form Factor: {desc['form_factor']:.6f}\n")
                f.write(f"Roundness: {desc['roundness']:.6f}\n")
                f.write(f"Compactness: {desc['compactness']:.6f}\n")
                f.write("-" * 30 + "\n")
        
        print(f"\nDescriptors saved to {filename}")
    
    def visualize_results(self, shapes):
        """Visualize original images, boundaries, and results"""
        fig, axes = plt.subplots(2, len(shapes), figsize=(15, 8))
        
        shape_names = list(shapes.keys())
        
        for i, (name, image) in enumerate(shapes.items()):
            # Original image
            axes[0, i].imshow(image, cmap='gray')
            axes[0, i].set_title(f'Original {name}')
            axes[0, i].axis('off')
            
            # Boundary
            if name in self.descriptors:
                boundary = self.descriptors[name]['boundary']
                axes[1, i].imshow(boundary, cmap='gray')
                axes[1, i].set_title(f'Boundary {name}')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()

# Main execution
def main():
    print("Image Processing Lab 5: Region Descriptors and Feature Matching")
    print("=" * 60)
    
    # Create region descriptor processor
    processor = RegionDescriptors()
    
    # Create sample shapes
    shapes = processor.create_sample_shapes()
    
    # Define train and test sets (using same shapes for demonstration)
    train_shapes = {
        'train_rectangle': shapes['rectangle'],
        'train_circle': shapes['circle'],
        'train_triangle': shapes['triangle']
    }
    
    test_shapes = {
        'test_rectangle': shapes['rectangle'],
        'test_circle': shapes['circle'], 
        'test_triangle': shapes['triangle']
    }
    
    # Process all images
    all_shapes = {**train_shapes, **test_shapes}
    
    for name, image in all_shapes.items():
        processor.process_image(image, name)
    
    # Match descriptors
    train_names = list(train_shapes.keys())
    test_names = list(test_shapes.keys())
    
    similarity_matrix = processor.match_descriptors(train_names, test_names)
    
    # Save results
    processor.save_descriptors()
    
    # Visualize results
    processor.visualize_results(shapes)
    
    print("\n" + "=" * 60)
    print("Lab 5 completed successfully!")
    print("Check 'descriptors.txt' for detailed results.")

if __name__ == "__main__":
    main()
