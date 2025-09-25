import numpy as np
import matplotlib.pyplot as plt

def plot_raw_numbers(input_file):
    with open(input_file, 'r') as f:
        numbers = [float(line.strip()) for line in f]
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(numbers)), numbers, 'b-', linewidth=1)
    
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Raw Values Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('raw_numbers.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Number of data points: {len(numbers)}")
    print(f"Average value: {np.mean(numbers):.4f}")
    print(f"Standard deviation: {np.std(numbers):.4f}")
    print(f"Min value: {min(numbers):.4f}")
    print(f"Max value: {max(numbers):.4f}")

if __name__ == "__main__":
    plot_raw_numbers('filtered_out.txt')