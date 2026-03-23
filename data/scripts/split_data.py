#!/usr/bin/env python3
"""Data splitting script

Merge general text data and recommendation data, then split into multiple files with 1000 samples each.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def find_parquet_files(directory: str, recursive: bool = True) -> List[str]:
    """Find all parquet files in the directory.

    Args:
        directory: Directory path
        recursive: Whether to recursively search subdirectories

    Returns:
        List of parquet file paths
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    pattern = "**/*.parquet" if recursive else "*.parquet"
    parquet_files = [str(p) for p in dir_path.glob(pattern) if p.is_file()]
    
    return sorted(parquet_files)


def load_all_parquet_files(file_paths: List[str], engine: str = 'pyarrow') -> pd.DataFrame:
    """Load and merge all parquet files.

    Args:
        file_paths: List of parquet file paths
        engine: Parquet engine, 'pyarrow' or 'fastparquet'

    Returns:
        Merged DataFrame
    """
    if not file_paths:
        logger.warning("No parquet files found")
        return pd.DataFrame()

    logger.info(f"Found {len(file_paths)} parquet files, starting to load...")

    dataframes = []
    for file_path in tqdm(file_paths, desc="Loading files"):
        try:
            df = pd.read_parquet(file_path, engine=engine)
            logger.debug(f"  Loaded {file_path}: {len(df)} rows")
            dataframes.append(df)
        except Exception as e:
            logger.error(f"  Failed to load {file_path}: {e}")
            continue

    if not dataframes:
        logger.warning("No files loaded successfully")
        return pd.DataFrame()

    # Merge all DataFrames
    logger.info("Merging all data...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Merge complete, total {len(combined_df)} rows")

    return combined_df


def split_dataframe(df: pd.DataFrame, max_rows: int, output_dir: str, prefix: str = "part") -> List[str]:
    """Split DataFrame into multiple files by fixed number of rows.

    Args:
        df: DataFrame to split
        max_rows: Maximum number of rows per file
        output_dir: Output directory
        prefix: Output file prefix

    Returns:
        List of output file paths
    """
    if len(df) == 0:
        logger.warning("DataFrame is empty, no need to split")
        return []

    if max_rows <= 0:
        raise ValueError(f"max_rows must be greater than 0, current value: {max_rows}")
    
    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Calculate number of files needed
    total_rows = len(df)
    num_chunks = (total_rows + max_rows - 1) // max_rows  # Round up
    logger.info(f"Splitting data into {num_chunks} files (max {max_rows} rows per file)")

    # Use fixed 5-digit format to ensure consistent file naming
    # Format: part-00000-of-00010.parquet
    num_digits = 5

    # Split and save
    output_files = []
    for chunk_idx in tqdm(range(num_chunks), desc="Splitting files"):
        start_idx = chunk_idx * max_rows
        end_idx = min(start_idx + max_rows, total_rows)

        # Extract data chunk
        chunk_df = df.iloc[start_idx:end_idx]

        # Generate output filename, format: part-00000-of-00010.parquet
        output_filename = f"{prefix}-{chunk_idx:0{num_digits}d}-of-{num_chunks:0{num_digits}d}.parquet"
        output_path = output_dir_path / output_filename

        # Save file
        chunk_df.to_parquet(
            output_path,
            engine='pyarrow',
            index=False,
            compression='snappy'
        )

        output_files.append(str(output_path))
        logger.debug(f"  Saved file {chunk_idx + 1}/{num_chunks}: {output_path} (rows {start_idx} to {end_idx - 1})")

    logger.info(f"Successfully split into {len(output_files)} files")
    return output_files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Merge general text data and recommendation data, then split into multiple files with 1000 samples each'
    )
    parser.add_argument(
        '--general_text_path',
        type=str,
        required=True,
        help='General text data path (directory or file)'
    )
    parser.add_argument(
        '--rec_data_path',
        type=str,
        required=True,
        help='Recommendation data path (directory or file)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory path'
    )
    parser.add_argument(
        '--max_rows',
        type=int,
        default=1000,
        help='Maximum number of rows per file (default: 1000)'
    )
    parser.add_argument(
        '--engine',
        choices=['pyarrow', 'fastparquet'],
        default='pyarrow',
        help='Parquet processing engine (default: pyarrow)'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not recursively search for files in subdirectories'
    )
    
    args = parser.parse_args()

    # Validate parameters
    if args.max_rows <= 0:
        logger.error(f"max_rows must be greater than 0, current value: {args.max_rows}")
        sys.exit(1)
    
    try:
        # 1. Find all parquet files
        logger.info("=" * 60)
        logger.info("Step 1: Finding general text data files...")
        general_text_path = Path(args.general_text_path)
        if general_text_path.is_file():
            general_text_files = [str(general_text_path)]
        else:
            general_text_files = find_parquet_files(
                args.general_text_path,
                recursive=not args.no_recursive
            )
        logger.info(f"Found {len(general_text_files)} general text files")

        logger.info("Step 2: Finding recommendation data files...")
        rec_data_path = Path(args.rec_data_path)
        if rec_data_path.is_file():
            rec_data_files = [str(rec_data_path)]
        else:
            rec_data_files = find_parquet_files(
                args.rec_data_path,
                recursive=not args.no_recursive
            )
        logger.info(f"Found {len(rec_data_files)} recommendation data files")
        
        # 2. Load all files
        logger.info("=" * 60)
        logger.info("Step 3: Loading general text data...")
        general_text_df = load_all_parquet_files(general_text_files, engine=args.engine)

        logger.info("Step 4: Loading recommendation data...")
        rec_data_df = load_all_parquet_files(rec_data_files, engine=args.engine)
        
        # 3. Merge data
        logger.info("=" * 60)
        logger.info("Step 5: Merging data...")
        if len(general_text_df) == 0 and len(rec_data_df) == 0:
            logger.error("No data loaded")
            sys.exit(1)

        if len(general_text_df) == 0:
            combined_df = rec_data_df
            logger.info("Using only recommendation data")
        elif len(rec_data_df) == 0:
            combined_df = general_text_df
            logger.info("Using only general text data")
        else:
            combined_df = pd.concat([general_text_df, rec_data_df], ignore_index=True)
            logger.info(f"Merge complete: general text {len(general_text_df)} rows + recommendation data {len(rec_data_df)} rows = total {len(combined_df)} rows")
        
        # 4. Split data
        logger.info("=" * 60)
        logger.info("Step 6: Splitting data...")
        output_files = split_dataframe(
            combined_df,
            max_rows=args.max_rows,
            output_dir=args.output_dir,
            prefix="part"
        )
        
        # 5. Generate file list JSON
        logger.info("=" * 60)
        logger.info("Step 7: Generating file list JSON...")
        output_dir_path = Path(args.output_dir)
        json_file_path = output_dir_path / "file_list.json"

        # Convert file paths to absolute paths (absolute paths are more reliable)
        file_list = [str(Path(f).absolute()) for f in output_files]

        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(file_list, f, indent=2, ensure_ascii=False)

        logger.info(f"File list saved to: {json_file_path} ({len(file_list)} files)")
        
        # 6. Output statistics
        logger.info("=" * 60)
        logger.info("Processing complete!")
        logger.info(f"Input files: general text {len(general_text_files)} files, recommendation data {len(rec_data_files)} files")
        logger.info(f"Total data rows: {len(combined_df)}")
        logger.info(f"Output files: {len(output_files)}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"File list JSON: {json_file_path}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Program execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

