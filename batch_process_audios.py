#!/usr/bin/env python3
"""
Batch processing script for audio files in stock folders.
This script processes all MP3 files in the audios directory structure.
"""

import os
from mp3_data_process import AudioProcessor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_stock_folders(base_dir: str = "audios", model_size: str = "base"):
    """
    Process all MP3 files in stock folders.
    
    Args:
        base_dir: Base directory containing stock folders
        model_size: Size of the Whisper model to use
    """
    processor = AudioProcessor(model_size=model_size)
    
    # Get all stock folders
    stock_folders = [f for f in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, f)) 
                    and not f.startswith('.')]
    
    total_folders = len(stock_folders)
    logger.info(f"Found {total_folders} stock folders to process")
    
    for idx, stock_folder in enumerate(stock_folders, 1):
        folder_path = os.path.join(base_dir, stock_folder)
        logger.info(f"Processing folder {idx}/{total_folders}: {stock_folder}")
        
        # Get all MP3 files in the folder
        mp3_files = [f for f in os.listdir(folder_path) 
                    if f.endswith('.mp3') and not f.startswith('.')]
        
        for mp3_file in mp3_files:
            input_path = os.path.join(folder_path, mp3_file)
            logger.info(f"Processing file: {mp3_file}")
            
            try:
                # Process the audio file
                result = processor.process_audio(input_path, folder_path)
                logger.info(f"Successfully processed {mp3_file}")
                
            except Exception as e:
                logger.error(f"Error processing {mp3_file}: {str(e)}")
                continue

def main():
    """Main function to process all stock folders."""
    try:
        process_stock_folders()
        logger.info("Batch processing completed successfully")
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 