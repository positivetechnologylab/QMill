import numpy as np
import struct
import os

def read_idx3_ubyte(filename):
    """
    Read IDX3-UBYTE file format
    
    Parameters:
        filename (str): Path to the IDX3-UBYTE file
        
    Returns:
        numpy.ndarray: Images as a 2D array (n_images x pixels)
    """
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        n_images = struct.unpack('>I', f.read(4))[0]
        n_rows = struct.unpack('>I', f.read(4))[0]
        n_cols = struct.unpack('>I', f.read(4))[0]
        
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(n_images, n_rows * n_cols)
        
        return images

def save_to_csv(images, output_filename):
    """
    Save images to CSV file
    
    Parameters:
        images (numpy.ndarray): Images as a 2D array
        output_filename (str): Path to save the CSV file
    """
    print("HERE", images[0])
    np.savetxt(output_filename, images, delimiter=',', fmt='%d')

def main():
    input_file = './data/train-images-idx3-ubyte'
    output_file = './data/fashion_mnist_images.csv'
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Reading {input_file}...")
    images = read_idx3_ubyte(input_file)
    
    print(f"Converting to CSV format...")
    save_to_csv(images, output_file)
    
    print(f"Successfully saved to {output_file}")
    print(f"Shape of the dataset: {images.shape}")

if __name__ == "__main__":
    main()