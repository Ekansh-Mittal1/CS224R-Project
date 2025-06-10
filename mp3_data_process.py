#!/usr/bin/env python3
"""
Audio processing module for financial transcript trading system.
This module handles MP3 file processing, metadata extraction, and transcription.
"""

import argparse
import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional
import whisper
from pydub import AudioSegment
import nltk
from nltk.tokenize import sent_tokenize
import ssl

# Fix SSL certificate issue
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Class for processing audio files and generating transcriptions.
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the AudioProcessor.
        
        Args:
            model_size: Size of the Whisper model to use
                      (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.model = None
        logger.info(f"Initializing AudioProcessor with {model_size} model")
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')
    
    def load_model(self):
        """Load the Whisper model."""
        if self.model is None:
            logger.info(f"Loading Whisper {self.model_size} model...")
            try:
                self.model = whisper.load_model(self.model_size)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {str(e)}")
                raise
    
    def process_audio(self, 
                     input_file: str, 
                     output_dir: Optional[str] = None) -> Dict:
        """
        Process an audio file and generate transcription.
        
        Args:
            input_file: Path to input audio file
            output_dir: Directory to save output files (default: same as input)
            
        Returns:
            Dictionary containing processing results
        """
        try:
            # Validate input file
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            # Set output directory
            if output_dir is None:
                output_dir = os.path.dirname(input_file)
            os.makedirs(output_dir, exist_ok=True)
            
            # Load and process audio
            logger.info(f"Processing audio file: {input_file}")
            audio = AudioSegment.from_mp3(input_file)
            
            # Extract metadata
            metadata = {
                'filename': os.path.basename(input_file),
                'duration': len(audio) / 1000.0,  # Convert to seconds
                'channels': audio.channels,
                'sample_width': audio.sample_width,
                'frame_rate': audio.frame_rate,
                'processed_date': datetime.now().isoformat()
            }
            
            # Generate transcription
            logger.info("Generating transcription...")
            self.load_model()
            
            # Transcribe using Whisper
            result = self.model.transcribe(input_file)
            
            # Process transcription
            transcript = result["text"]
            segments = result["segments"]
            
            # Save results
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            
            # Save metadata
            metadata_file = os.path.join(output_dir, f"{base_name}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save full transcript
            transcript_file = os.path.join(output_dir, f"{base_name}_transcript.txt")
            with open(transcript_file, 'w') as f:
                f.write(transcript)
            
            # Save segmented transcript with timestamps
            segments_file = os.path.join(output_dir, f"{base_name}_segments.txt")
            with open(segments_file, 'w') as f:
                for segment in segments:
                    start_time = datetime.fromtimestamp(segment['start']).strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"[{start_time}] {segment['text']}\n")
            
            logger.info(f"Processing complete. Results saved to {output_dir}")
            
            return {
                'metadata': metadata,
                'transcript': transcript,
                'segments': segments,
                'output_files': {
                    'metadata': metadata_file,
                    'transcript': transcript_file,
                    'segments': segments_file
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            raise


def main():
    """Main function to process audio files."""
    parser = argparse.ArgumentParser(description='Process audio files and generate transcriptions.')
    parser.add_argument('input_file', help='Path to input audio file')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('-m', '--model', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       default='base',
                       help='Whisper model size to use')
    
    args = parser.parse_args()
    
    try:
        processor = AudioProcessor(model_size=args.model)
        result = processor.process_audio(args.input_file, args.output)
        
        # Print summary
        print("\n" + "="*80)
        print("PROCESSING SUMMARY")
        print("="*80)
        print(f"Input File: {args.input_file}")
        print(f"Duration: {result['metadata']['duration']:.2f} seconds")
        print(f"Output Directory: {os.path.dirname(result['output_files']['transcript'])}")
        print("\nOutput Files:")
        for key, path in result['output_files'].items():
            print(f"- {key}: {path}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
