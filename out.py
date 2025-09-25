import sys

def filter_lines(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if not line.startswith(('Box', 'Total')):
                f_out.write(line)

if __name__ == "__main__":
    filter_lines('out.txt', 'filtered_out.txt')