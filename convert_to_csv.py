import pandas as pd

def convert_md_to_csv(input_file, output_file):
    # Read the markdown file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Find where the table starts (skip the title and blank lines)
    table_start = 0
    for i, line in enumerate(lines):
        if line.startswith('|'):
            table_start = i
            break
    
    # Extract table lines
    table_lines = [line.strip() for line in lines[table_start:] if line.strip()]
    
    # Remove the header separator line
    table_lines.pop(1)
    
    # Process each line to clean up the markdown table formatting
    cleaned_lines = []
    for line in table_lines:
        # Remove leading/trailing |
        line = line.strip('|')
        # Split by | and strip whitespace from each cell
        cells = [cell.strip() for cell in line.split('|')]
        cleaned_lines.append(cells)
    
    # Create DataFrame
    df = pd.DataFrame(cleaned_lines[1:], columns=cleaned_lines[0])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Successfully converted {input_file} to {output_file}")

if __name__ == "__main__":
    input_file = "automated_trades.md"
    output_file = "automated_trades.csv"
    convert_md_to_csv(input_file, output_file) 