#!/usr/bin/env python3
"""
Generate test images in PPM format for image processing tests
"""

import numpy as np
import sys

def save_ppm(filename, image):
    """Save image as PPM (P6 format)"""
    height, width, channels = image.shape
    with open(filename, 'wb') as f:
        f.write(f'P6\n{width} {height}\n255\n'.encode())
        f.write(image.tobytes())
    print(f"Saved: {filename} ({width}x{height})")

def generate_gradient_image(width, height):
    """Generate a colorful gradient image"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            image[i, j, 0] = int(255 * i / height)      # Red gradient
            image[i, j, 1] = int(255 * j / width)        # Green gradient
            image[i, j, 2] = int(255 * (1 - i/height))  # Blue gradient
    
    return image

def generate_checkerboard(width, height, square_size=50):
    """Generate a checkerboard pattern"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                image[i, j] = [255, 255, 255]  # White
            else:
                image[i, j] = [0, 0, 0]        # Black
    
    return image

def generate_circles_image(width, height):
    """Generate image with colored circles"""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Draw some circles
    circles = [
        (width//4, height//4, 80, [255, 0, 0]),      # Red
        (3*width//4, height//4, 60, [0, 255, 0]),    # Green
        (width//2, 3*height//4, 100, [0, 0, 255]),   # Blue
    ]
    
    for cx, cy, radius, color in circles:
        for i in range(height):
            for j in range(width):
                dist = np.sqrt((i - cy)**2 + (j - cx)**2)
                if dist < radius:
                    image[i, j] = color
    
    return image

def generate_noise_image(width, height):
    """Generate random noise image"""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

if __name__ == "__main__":
    # Different sizes for testing scalability
    sizes = [
        (512, 512, "small"),
        (1024, 1024, "medium"),
        (2048, 2048, "large"),
        (4096, 4096, "xlarge")
    ]
    
    # Generate different test images
    for width, height, size_name in sizes:
        # Gradient
        img = generate_gradient_image(width, height)
        save_ppm(f"test_gradient_{size_name}.ppm", img)
        
        # Only generate other patterns for smaller sizes
        if size_name in ["small", "medium"]:
            # Checkerboard
            img = generate_checkerboard(width, height)
            save_ppm(f"test_checkerboard_{size_name}.ppm", img)
            
            # Circles
            img = generate_circles_image(width, height)
            save_ppm(f"test_circles_{size_name}.ppm", img)
    
    print("\nTest images generated!")
    print("You can now use these with your MPI program:")
    print("  mpirun -np 4 ./image_proc_mpi.exe test_gradient_medium.ppm output.ppm blur")