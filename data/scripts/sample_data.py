#!/usr/bin/env python3
"""Data Sampling Script

Sample specified number of samples from one or more paths (directories or files) containing parquet files,
and save as a single parquet file.
"""

import argparse
import logging
import random
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


def collect_parquet_files(input_paths: List[str], recursive: bool = True) -> List[str]:
    """Collect all parquet file paths.

    Args:
        input_paths: List of input paths (can be files or directories)
        recursive: Whether to recursively search subdirectories

    Returns:
        List of parquet file paths
    """
    all_files = []
    
    for input_path in input_paths:
        path = Path(input_path)

        if not path.exists():
            logger.warning(f"Path does not exist, skipping: {input_path}")
            continue

        if path.is_file():
            if path.suffix.lower() == '.parquet':
                all_files.append(str(path))
            else:
                logger.warning(f"Not a parquet file, skipping: {input_path}")
        elif path.is_dir():
            files = find_parquet_files(str(path), recursive=recursive)
            all_files.extend(files)
        else:
            logger.warning(f"Unknown path type, skipping: {input_path}")

    return sorted(list(set(all_files)))  # Remove duplicates and sort


def load_all_parquet_files(file_paths: List[str], engine: str = 'pyarrow') -> pd.DataFrame:
    """Load all parquet files and merge them.

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
    logger.info(f"Merge completed, total {len(combined_df)} rows")

    return combined_df


def sample_dataframe(df: pd.DataFrame, num_samples: int, seed: int = None) -> pd.DataFrame:
    """Sample specified number of samples from DataFrame.

    Args:
        df: DataFrame to sample from
        num_samples: Number of samples
        seed: Random seed

    Returns:
        Sampled DataFrame
    """
    if len(df) == 0:
        logger.warning("DataFrame is empty, cannot sample")
        return pd.DataFrame()

    if num_samples <= 0:
        raise ValueError(f"num_samples must be greater than 0, current value: {num_samples}")

    total_rows = len(df)

    if num_samples >= total_rows:
        logger.warning(f"Sample size ({num_samples}) is greater than or equal to total rows ({total_rows}), returning all data")
        return df.copy()
    
    # Set random seed
    if seed is not None:
        random.seed(seed)
        logger.info(f"Using random seed: {seed}")

    # Random sampling
    logger.info(f"Sampling {num_samples} rows from {total_rows} rows...")
    sampled_indices = random.sample(range(total_rows), num_samples)
    sampled_df = df.iloc[sampled_indices].copy()

    logger.info(f"Sampling completed, total {len(sampled_df)} rows")

    return sampled_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Sample specified number of samples from one or more paths containing parquet files, and save as a single parquet file'
    )
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        required=True,
        help='Input paths (can be files or directories), multiple paths can be specified'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output parquet file path'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        required=True,
        help='Number of samples'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (optional)'
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
    if args.num_samples <= 0:
        logger.error(f"num_samples must be greater than 0, current value: {args.num_samples}")
        sys.exit(1)
    
    try:
        # 1. Collect all parquet files
        logger.info("=" * 60)
        logger.info("Step 1: Collecting parquet files...")
        parquet_files = collect_parquet_files(
            args.input,
            recursive=not args.no_recursive
        )

        if not parquet_files:
            logger.error("No parquet files found")
            sys.exit(1)

        logger.info(f"Found {len(parquet_files)} parquet files")
        
        # 2. Load all files
        logger.info("=" * 60)
        logger.info("Step 2: Loading parquet files...")
        combined_df = load_all_parquet_files(parquet_files, engine=args.engine)

        if len(combined_df) == 0:
            logger.error("No data loaded")
            sys.exit(1)
        
        # 3. Sample data
        logger.info("=" * 60)
        logger.info("Step 3: Sampling data...")
        sampled_df = sample_dataframe(
            combined_df,
            num_samples=args.num_samples,
            seed=args.seed
        )

        if len(sampled_df) == 0:
            logger.error("Sampled data is empty")
            sys.exit(1)
        
        # 4. Save results
        logger.info("=" * 60)
        logger.info("Step 4: Saving results...")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sampled_df.to_parquet(
            output_path,
            engine='pyarrow',
            index=False,
            compression='snappy'
        )

        logger.info(f"Results saved to: {output_path}")

        # 5. Output statistics
        logger.info("=" * 60)
        logger.info("Processing completed!")
        logger.info(f"Input files: {len(parquet_files)}")
        logger.info(f"Original data rows: {len(combined_df)}")
        logger.info(f"Sampled rows: {len(sampled_df)}")
        logger.info(f"Output file: {output_path}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Program execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

