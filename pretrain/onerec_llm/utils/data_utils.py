"""Data loading utilities for parquet files and HDFS."""

import hashlib
import os
import subprocess
import time
import traceback
from typing import Optional

import numpy as np
import pyarrow.parquet as pq

from onerec_llm.utils.worker_utils import get_worker_info
from onerec_llm.utils.distributed import get_world_size_and_rank


def calculate_text_hash(text):
    """Calculate SHA-256 hash of text.
    
    Args:
        text: Input text string
        
    Returns:
        Hexadecimal hash string
    """
    hash_object = hashlib.sha256()
    hash_object.update(text.encode('utf-8'))
    return hash_object.hexdigest()


def shell_hdfs_ls(source_dir):
    """List files in HDFS directory.
    
    Args:
        source_dir: HDFS directory path
        
    Returns:
        list: List of file paths starting with 'viewfs://'
    """
    try:
        command = f"hdfs dfs -ls {source_dir}"
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        files = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) > 0 and parts[-1].startswith('viewfs://'):
                files.append(parts[-1])
        return files

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {traceback.format_exc()}")
        return []


class FakeParquetFileFromFastParquetFile:
    """Wrapper for fastparquet ParquetFile to match pyarrow interface."""
    
    def __init__(self, fast_parquet_file):
        # Package version: mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pip3 install fastparquet==2024.2.0"
        from fastparquet import ParquetFile
        self.fast_parquet_file = fast_parquet_file

        # Put file opening logic first to prevent failure if file is deleted
        self.res = ParquetFile(self.fast_parquet_file)
        self.res.num_rows = len(self.res.to_pandas())
        self.num_row_groups = 1

    def read_row_group(self, i):
        assert i == 0
        return self.res


def load_parquet_file(
    file_path: str,
    retry: int = 5,
    max_cache_files: int = 500,
    parquet_backend: str = 'pyarrow',
    cache_dir: Optional[str] = None,
    hadoop_cmd: Optional[str] = None
) -> pq.ParquetFile:
    """Load a parquet file from local path or HDFS.
    
    This function handles two types of paths:
    1. HDFS paths (viewfs:// or hdfs://): Downloads to cache and loads from cache
    2. Local paths: Directly loads from the path
    
    Args:
        file_path: Path to parquet file (can be local path or HDFS path)
        retry: Number of retries when HDFS download fails
        max_cache_files: Maximum number of files to keep in cache
        parquet_backend: Parquet backend, 'fast_parquet' or 'pyarrow'
        cache_dir: Cache directory path (default: /code/dataset_cache/{worker_id}_{rank_id})
        hadoop_cmd: Hadoop command path (default: /home/hadoop/software/hadoop/bin/hadoop)
        
    Returns:
        Loaded parquet file object
        
    Raises:
        ValueError: If parquet_backend is invalid
        FileNotFoundError: If file cannot be found or downloaded after retries
    """
    if parquet_backend not in ["fast_parquet", "pyarrow"]:
        raise ValueError(f"Invalid parquet_backend: {parquet_backend}. Must be 'fast_parquet' or 'pyarrow'")
    
    # Check if it's an HDFS path
    is_hdfs_path = file_path.startswith(('viewfs://', 'hdfs://'))
    
    if is_hdfs_path:
        # HDFS path: use cache and download logic
        return _load_parquet_from_hdfs(
            file_path, retry, max_cache_files, parquet_backend, cache_dir, hadoop_cmd
        )
    else:
        # Local path: directly load (even if os.path.exists returns False,
        # some file systems may support direct access)
        try:
            return _load_parquet_from_path(file_path, parquet_backend)
        except Exception as e:
            # If direct load fails and file doesn't exist, provide clear error
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Local file not found: {file_path}") from e
            raise


def _load_parquet_from_hdfs(
    file_path: str,
    retry: int,
    max_cache_files: int,
    parquet_backend: str,
    cache_dir: Optional[str],
    hadoop_cmd: Optional[str]
) -> pq.ParquetFile:
    """Load parquet file from HDFS using cache mechanism."""
    # Setup cache directory
    # If cache_dir is None or empty string, use default cache directory
    if not cache_dir:
        worker_id = get_worker_info()[0]
        rank_id = get_world_size_and_rank()[1]
        cache_dir = f'/code/dataset_cache/{worker_id}_{rank_id}'
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache file path
    filename = os.path.basename(file_path)
    file_hash = calculate_text_hash(file_path)
    cache_path = os.path.join(cache_dir, f"{file_hash}_{filename}")
    
    # Try to load from cache first
    if os.path.exists(cache_path):
        try:
            return _load_parquet_from_path(cache_path, parquet_backend)
        except Exception as e:
            # Cache file might be corrupted, remove it and re-download
            print(f"Warning: Cached file {cache_path} is corrupted, removing: {e}")
            try:
                os.remove(cache_path)
            except:
                pass
    
    # Download from HDFS with retry
    if hadoop_cmd is None:
        hadoop_cmd = '/home/hadoop/software/hadoop/bin/hadoop'
    
    last_error = None
    for attempt in range(retry):
        try:
            # Clean cache if needed before downloading
            _clean_cache_if_needed(cache_dir, max_cache_files)
            
            # Download from HDFS
            _download_from_hdfs(file_path, cache_path, hadoop_cmd)
            
            # Load downloaded file
            return _load_parquet_from_path(cache_path, parquet_backend)
            
        except Exception as e:
            last_error = e
            if attempt < retry - 1:
                # Exponential backoff with jitter
                wait_time = 2 + np.random.randint(0, 5) + attempt
                print(f"Download attempt {attempt + 1}/{retry} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"All {retry} download attempts failed for {file_path}")
    
    # All retries failed
    raise FileNotFoundError(
        f"Failed to load parquet file from HDFS after {retry} attempts. "
        f"HDFS path: {file_path}, Cache: {cache_path}, Error: {last_error}"
    )


def _load_parquet_from_path(file_path: str, parquet_backend: str) -> pq.ParquetFile:
    """Load parquet file from given path."""
    if parquet_backend == 'pyarrow':
        return pq.ParquetFile(file_path)
    else:
        return FakeParquetFileFromFastParquetFile(file_path)


def _clean_cache_if_needed(cache_dir: str, max_cache_files: int):
    """Clean old cache files if cache exceeds max_cache_files."""
    try:
        files = [
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if os.path.isfile(os.path.join(cache_dir, f))
        ]
        
        if len(files) <= max_cache_files:
            return
        
        # Sort by creation time and remove oldest half
        files.sort(key=os.path.getctime)
        files_to_remove = files[:len(files) - max_cache_files // 2]
        
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                print(f"Removed old cached file: {file_path}")
            except Exception as e:
                print(f"Failed to remove cached file {file_path}: {e}")
    except Exception as e:
        print(f"Warning: Failed to clean cache: {e}")


def _download_from_hdfs(hdfs_path: str, local_path: str, hadoop_cmd: str):
    """Download file from HDFS to local path."""
    cmd = [hadoop_cmd, 'fs', '-get', hdfs_path, local_path]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"HDFS download failed. Command: {' '.join(cmd)}, "
            f"Return code: {result.returncode}, "
            f"Error: {result.stderr}"
        )
    
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Downloaded file not found at {local_path}")

