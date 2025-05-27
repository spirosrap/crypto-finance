import pandas as pd
import os
import argparse

def generate_margin_files(input_file, margin_values):
    """
    Generate new CSV files with different margin values from the input file.
    
    Args:
        input_file (str): Path to the input CSV file
        margin_values (list): List of margin values to generate files for
    """
    try:
        # Read the original CSV file
        df = pd.read_csv(input_file)
        
        # Generate a new file for each margin value
        for margin in margin_values:
            # Create a copy of the dataframe
            df_new = df.copy()
            
            # Update the margin value
            df_new['Margin'] = margin
            
            # Generate output filename
            output_file = f"automated_trades_{int(margin)}.csv"
            
            # Save to CSV
            df_new.to_csv(output_file, index=False)
            print(f"Created {output_file} with margin {margin}")
            
    except FileNotFoundError:
        print(f"Error: Could not find the input file '{input_file}'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate CSV files with different margin values')
    parser.add_argument('-i', '--input', 
                      default="automated_trades.csv",
                      help='Input CSV file (default: automated_trades.csv)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Margin values to generate
    margin_values = [500, 1000, 2000, 5000]
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
    else:
        generate_margin_files(args.input, margin_values) 