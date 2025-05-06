#!/usr/bin/env python3
"""
Feather to JSON Converter

A tool to convert Feather files (used by Freqtrade) to JSON format.
"""

import pandas as pd
import json
import argparse
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feather_to_json')

def convert_feather_to_json(input_file, output_file=None, orient='records', date_format='iso', indent=2):
    """
    Convert a Feather file to JSON format.
    
    Parameters:
    -----------
    input_file : str
        Path to the input Feather file.
    output_file : str, optional
        Path to the output JSON file. If None, will use the same name as input with .json extension.
    orient : str, optional
        The format of the JSON string. Default is 'records'.
        Options: 'split', 'records', 'index', 'columns', 'values', 'table'.
    date_format : str, optional
        Format for dates in JSON output. Default is 'iso'.
    indent : int, optional
        Indentation level for JSON formatting. Default is 2.
    
    Returns:
    --------
    str
        Path to the created JSON file.
    """
    try:
        logger.info(f"Reading Feather file: {input_file}")
        df = pd.read_feather(input_file)
        
        # If no output file specified, use input name with .json extension
        if output_file is None:
            input_path = Path(input_file)
            output_file = str(input_path.with_suffix('.json'))
        
        logger.info(f"Dataframe shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Convert datetime columns to string
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        # Convert the DataFrame to JSON
        logger.info(f"Converting to JSON with orient={orient}")
        json_data = df.to_json(orient=orient, date_format=date_format)
        
        # Load the JSON data to format it nicely
        parsed_json = json.loads(json_data)
        
        # Write the formatted JSON to the output file
        logger.info(f"Writing JSON to file: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(parsed_json, f, indent=indent)
        
        logger.info(f"Conversion completed! JSON file saved to: {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error converting file: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Convert Feather files to JSON format.')
    parser.add_argument('input', help='Input Feather file path')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('--orient', default='records', 
                        choices=['split', 'records', 'index', 'columns', 'values', 'table'],
                        help='JSON orientation format (default: records)')
    parser.add_argument('--indent', type=int, default=2, 
                        help='JSON indentation level (default: 2)')
    parser.add_argument('--no-indent', action='store_true', 
                        help='Disable JSON indentation to reduce file size')
    
    args = parser.parse_args()
    
    indent = None if args.no_indent else args.indent
    
    try:
        output_file = convert_feather_to_json(
            args.input, 
            args.output, 
            orient=args.orient, 
            indent=indent
        )
        print(f"Successfully converted {args.input} to {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
