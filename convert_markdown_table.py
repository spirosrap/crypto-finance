import pandas as pd
import argparse
import os

def get_default_output(input_file, format='csv'):
    # Remove .md extension if present and add appropriate extension
    base_name = input_file[:-3] if input_file.endswith('.md') else input_file
    return f"{base_name}.{'xlsx' if format == 'excel' else 'csv'}"

def convert_md_to_table(input_file, output_file, format='csv'):
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
    
    # Save to CSV or Excel based on format
    if format == 'excel':
        df.to_excel(output_file, index=False)
    else:  # csv
        df.to_csv(output_file, index=False)
    
    print(f"Successfully converted {input_file} to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert markdown table to CSV or Excel')
    parser.add_argument('--input', dest='input_file', default='automated_trades.md',
                      help='Input markdown file (default: automated_trades.md)')
    parser.add_argument('--output', dest='output_file',
                      help='Output file (CSV or Excel). If not specified, uses input filename with appropriate extension')
    parser.add_argument('--format', choices=['csv', 'excel'], default='csv',
                      help='Output format (default: csv)')
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if not args.output_file:
        args.output_file = get_default_output(args.input_file, args.format)
    # Validate file extensions
    elif args.format == 'excel' and not args.output_file.endswith(('.xlsx', '.xls')):
        args.output_file += '.xlsx'
    elif args.format == 'csv' and not args.output_file.endswith('.csv'):
        args.output_file += '.csv'
    
    convert_md_to_table(args.input_file, args.output_file, args.format) 