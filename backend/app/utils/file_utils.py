"""
File Utilities
Helper functions for file operations and validation
"""

import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import hashlib
import mimetypes

logger = logging.getLogger(__name__)

async def save_upload_file(content: bytes, filepath: Path) -> bool:
    """
    Save uploaded file content to disk
    
    Args:
        content: File content as bytes
        filepath: Path where to save the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(filepath, 'wb') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save file {filepath}: {str(e)}")
        return False

def get_file_info(filepath: Path) -> Dict[str, Any]:
    """
    Get comprehensive file information
    
    Args:
        filepath: Path to the file
        
    Returns:
        Dictionary with file information
    """
    try:
        stat = filepath.stat()
        
        info = {
            "filename": filepath.name,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "extension": filepath.suffix.lower(),
            "mime_type": mimetypes.guess_type(str(filepath))[0],
            "created_at": stat.st_ctime,
            "modified_at": stat.st_mtime,
            "is_readable": os.access(filepath, os.R_OK),
            "is_writable": os.access(filepath, os.W_OK)
        }
        
        # Calculate file hash for integrity checking
        info["md5_hash"] = calculate_file_hash(filepath)
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get file info for {filepath}: {str(e)}")
        return {}

def calculate_file_hash(filepath: Path, algorithm: str = "md5") -> str:
    """
    Calculate file hash for integrity checking
    
    Args:
        filepath: Path to the file
        algorithm: Hash algorithm to use
        
    Returns:
        Hex digest of the file hash
    """
    try:
        hash_obj = hashlib.new(algorithm)
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
        
    except Exception as e:
        logger.error(f"Failed to calculate hash for {filepath}: {str(e)}")
        return ""

def validate_file_type(filepath: Path, allowed_extensions: List[str]) -> Tuple[bool, str]:
    """
    Validate file type against allowed extensions
    
    Args:
        filepath: Path to the file
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    extension = filepath.suffix.lower()
    
    if extension not in allowed_extensions:
        return False, f"File type {extension} not allowed. Allowed types: {allowed_extensions}"
    
    return True, ""

def validate_file_size(filepath: Path, max_size_bytes: int) -> Tuple[bool, str]:
    """
    Validate file size against maximum allowed size
    
    Args:
        filepath: Path to the file
        max_size_bytes: Maximum allowed file size in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        file_size = filepath.stat().st_size
        
        if file_size > max_size_bytes:
            max_size_mb = max_size_bytes / (1024 * 1024)
            actual_size_mb = file_size / (1024 * 1024)
            return False, f"File size {actual_size_mb:.1f}MB exceeds maximum allowed size {max_size_mb:.1f}MB"
        
        return True, ""
        
    except Exception as e:
        return False, f"Failed to check file size: {str(e)}"

def clean_filename(filename: str) -> str:
    """
    Clean filename by removing/replacing invalid characters
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    cleaned = filename
    
    for char in invalid_chars:
        cleaned = cleaned.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    cleaned = cleaned.strip(' .')
    
    # Ensure filename is not empty
    if not cleaned:
        cleaned = "unnamed_file"
    
    return cleaned

def ensure_directory_exists(directory: Path) -> bool:
    """
    Ensure directory exists, create if necessary
    
    Args:
        directory: Path to directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {str(e)}")
        return False

def cleanup_old_files(directory: Path, max_age_days: int = 7) -> int:
    """
    Clean up old files in a directory
    
    Args:
        directory: Directory to clean
        max_age_days: Maximum age of files to keep
        
    Returns:
        Number of files deleted
    """
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        deleted_count = 0
        
        for filepath in directory.iterdir():
            if filepath.is_file():
                file_age = current_time - filepath.stat().st_mtime
                
                if file_age > max_age_seconds:
                    try:
                        filepath.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old file: {filepath}")
                    except Exception as e:
                        logger.warning(f"Failed to delete old file {filepath}: {str(e)}")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Failed to cleanup old files in {directory}: {str(e)}")
        return 0

def get_directory_size(directory: Path) -> int:
    """
    Calculate total size of all files in a directory
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
    """
    try:
        total_size = 0
        
        for filepath in directory.rglob('*'):
            if filepath.is_file():
                total_size += filepath.stat().st_size
        
        return total_size
        
    except Exception as e:
        logger.error(f"Failed to calculate directory size for {directory}: {str(e)}")
        return 0

def safe_copy_file(source: Path, destination: Path) -> bool:
    """
    Safely copy a file with error handling
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(source, destination)
        
        # Verify copy was successful
        if destination.exists() and destination.stat().st_size == source.stat().st_size:
            return True
        else:
            logger.error(f"File copy verification failed: {source} -> {destination}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to copy file {source} -> {destination}: {str(e)}")
        return False

def safe_move_file(source: Path, destination: Path) -> bool:
    """
    Safely move a file with error handling
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file
        shutil.move(str(source), str(destination))
        
        # Verify move was successful
        if destination.exists() and not source.exists():
            return True
        else:
            logger.error(f"File move verification failed: {source} -> {destination}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to move file {source} -> {destination}: {str(e)}")
        return False

def get_available_disk_space(directory: Path) -> int:
    """
    Get available disk space for a directory
    
    Args:
        directory: Directory path
        
    Returns:
        Available space in bytes
    """
    try:
        stat = shutil.disk_usage(directory)
        return stat.free
    except Exception as e:
        logger.error(f"Failed to get disk space for {directory}: {str(e)}")
        return 0

def compress_file(filepath: Path, compression_type: str = "gzip") -> Optional[Path]:
    """
    Compress a file using specified compression
    
    Args:
        filepath: Path to file to compress
        compression_type: Type of compression (gzip, zip)
        
    Returns:
        Path to compressed file or None if failed
    """
    try:
        if compression_type == "gzip":
            import gzip
            compressed_path = filepath.with_suffix(filepath.suffix + '.gz')
            
            with open(filepath, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            return compressed_path
            
        elif compression_type == "zip":
            import zipfile
            compressed_path = filepath.with_suffix('.zip')
            
            with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(filepath, filepath.name)
            
            return compressed_path
        
        else:
            logger.error(f"Unsupported compression type: {compression_type}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to compress file {filepath}: {str(e)}")
        return None
