# Trading agent that uses PPO to make trading decisions based on transcript analysis.
# Supports multiple transformer models for text encoding.

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel,
    LongformerTokenizer, LongformerModel,
    AutoModelForSeq2SeqLM  # For LongT5
)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gym
from gym import spaces
import logging
from sklearn.model_selection import train_test_split
import random
import time
from datetime import datetime, timedelta
from functools import partial
import multiprocessing
import platform
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cpu")
logger.info("Using CPU device")

# Cache for transformer models and tokenizers
model_cache = {}
tokenizer_cache = {}

class TransformerEncoder(nn.Module):
    """Neural network for encoding text using different transformer models."""
    
    def __init__(self, model_name, hidden_size=256):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        
        # Get the base model
        self.transformer = get_transformer_model(model_name)
        self.embedding_size = 768  # Standard size for all models
        
        # Add projection layer
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            # Get model outputs
            if self.model_name == "longt5":
                # For LongT5, we need to provide decoder inputs
                decoder_input_ids = torch.ones_like(input_ids)  # Use ones instead of zeros
                outputs = self.transformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    return_dict=True
                )
                # Use encoder's last hidden state
                pooled = outputs.encoder_last_hidden_state[:, 0, :]
            else:
                outputs = self.transformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                # Handle different output types
                if self.model_name == "finbert":
                    # For FinBERT, use the last hidden state of the [CLS] token
                    pooled = outputs.last_hidden_state[:, 0, :]
                elif self.model_name == "longformer":
                    # For Longformer, use the last hidden state of the [CLS] token
                    pooled = outputs.last_hidden_state[:, 0, :]
                else:
                    # Fallback to last hidden state
                    pooled = outputs.last_hidden_state[:, 0, :]
            
            # Project and convert to numpy
            projected = self.fc(pooled)
            return projected.detach().numpy()

def get_transformer_model(model_name):
    """Get or create cached transformer model."""
    global model_cache
    if model_name not in model_cache:
        try:
            if model_name == "finbert":
                model_cache[model_name] = AutoModel.from_pretrained("ProsusAI/finbert", return_dict=True)
            elif model_name == "longformer":
                model_cache[model_name] = LongformerModel.from_pretrained("allenai/longformer-base-4096", return_dict=True)
            elif model_name == "longt5":
                # Use the encoder-decoder version for LongT5
                model_cache[model_name] = AutoModelForSeq2SeqLM.from_pretrained(
                    "google/long-t5-tglobal-base",
                    return_dict=True
                )
            else:
                raise ValueError(f"Unknown model name: {model_name}")
            model_cache[model_name].eval()
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    return model_cache[model_name]

def get_tokenizer(model_name):
    """Get or create cached tokenizer."""
    global tokenizer_cache
    if model_name not in tokenizer_cache:
        try:
            if model_name == "finbert":
                tokenizer_cache[model_name] = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            elif model_name == "longformer":
                tokenizer_cache[model_name] = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
            elif model_name == "longt5":
                tokenizer_cache[model_name] = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
            else:
                raise ValueError(f"Unknown model name: {model_name}")
        except Exception as e:
            logger.error(f"Error loading tokenizer {model_name}: {str(e)}")
            raise
    return tokenizer_cache[model_name]

class TimeTracker:
    """Class to track training and testing times."""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}
        
    def start(self):
        """Start timing."""
        self.start_time = time.time()
        self.checkpoints = {}
        
    def checkpoint(self, name):
        """Record a checkpoint."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        self.checkpoints[name] = time.time() - self.start_time
        
    def get_elapsed(self):
        """Get total elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        return time.time() - self.start_time
    
    def format_time(self, seconds):
        """Format seconds into readable time."""
        return str(timedelta(seconds=int(seconds)))
    
    def print_summary(self):
        """Print timing summary."""
        logger.info("\nTiming Summary:")
        logger.info(f"Total elapsed time: {self.format_time(self.get_elapsed())}")
        for name, elapsed in self.checkpoints.items():
            logger.info(f"{name}: {self.format_time(elapsed)}")

class TranscriptDataset(Dataset):
    """Dataset for loading transcript segments."""
    
    def __init__(self, transcript_file, model_name, max_length=512):
        self.data = pd.read_csv(transcript_file, sep='\t')
        self.tokenizer = get_tokenizer(model_name)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Text']
        time = self.data.iloc[idx]['Time (seconds)']
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'time': torch.tensor(time, dtype=torch.float)
        }

class TradingEnv(gym.Env):
    """Trading environment for the PPO agent."""
    
    def __init__(self, merged_file, model_name="finbert", initial_balance=10000):
        super().__init__()
        
        # Load merged data
        self.data = pd.read_csv(merged_file)
        self.data['ts_event'] = pd.to_datetime(self.data['ts_event'])
        
        # Pre-compute price data for faster access
        self.price_data = self.data.set_index('ts_event')
        
        # Initialize state
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.current_step = 0
        self.last_price = None
        
        # Initialize text encoder and tokenizer
        self.text_encoder = TransformerEncoder(model_name)
        self.text_encoder.eval()
        self.tokenizer = get_tokenizer(model_name)
        
        # Pre-tokenize all texts for faster access
        self.tokenized_texts = []
        for text in self.data['Text']:
            encoding = self.tokenizer(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.tokenized_texts.append(encoding)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: Buy, 1: Sell, 2: Hold
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(256 + 7,),  # Added position and balance to observation
            dtype=np.float32
        )
    
    def seed(self, seed=None):
        """Set the random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        return [seed]
    
    def _get_observation(self):
        # Get current data row
        current_row = self.data.iloc[self.current_step]
        
        # Get current price data using pre-computed index
        current_time = current_row['ts_event']
        try:
            current_price = self.price_data.loc[current_time:current_time + pd.Timedelta(seconds=1)].iloc[0]
        except (KeyError, IndexError):
            current_price = self.price_data.iloc[-1]
        
        # Use pre-tokenized text
        text_embedding = self.text_encoder(
            self.tokenized_texts[self.current_step]['input_ids'],
            self.tokenized_texts[self.current_step]['attention_mask']
        )
        
        # Ensure text_embedding is 1D
        if len(text_embedding.shape) > 1:
            text_embedding = text_embedding.squeeze()
        
        # Combine features
        price_features = np.array([
            current_price['open'],
            current_price['high'],
            current_price['low'],
            current_price['close'],
            current_price['volume'],
            self.position,  # Add position to observation
            self.balance / self.initial_balance  # Normalized balance
        ], dtype=np.float32)
        
        # Ensure both arrays are 1D before concatenation
        return np.concatenate([text_embedding, price_features])
    
    def _calculate_reward(self, action, current_price):
        # Calculate price change
        if self.last_price is None:
            price_change = 0
        else:
            price_change = (current_price - self.last_price) / self.last_price
        
        # Base reward on action and price change
        if action == 0:  # Buy
            reward = price_change * 2  # Amplify rewards for correct buy decisions
        elif action == 1:  # Sell
            reward = -price_change * 2  # Amplify rewards for correct sell decisions
        else:  # Hold
            reward = 0
        
        # Add position-based reward
        if self.position > 0:
            reward += price_change * self.position  # Reward for holding during price increases
        elif self.position < 0:
            reward -= price_change * abs(self.position)  # Reward for shorting during price decreases
        
        # Penalize excessive holding
        if action == 2 and self.position == 0:
            reward -= 0.001  # Small penalty for holding with no position
        
        return reward
    
    def step(self, action):
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        if action == 0:  # Buy
            if self.balance >= current_price:
                self.position += 1
                self.balance -= current_price
        elif action == 1:  # Sell
            if self.position > 0:
                self.position -= 1
                self.balance += current_price
        
        # Calculate reward
        reward = self._calculate_reward(action, current_price)
        
        # Update last price
        self.last_price = current_price
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Get new observation
        obs = self._get_observation()
        
        # Additional info
        info = {
            'balance': self.balance,
            'position': self.position,
            'current_price': current_price,
            'portfolio_value': self.balance + (self.position * current_price)
        }
        
        return obs, reward, done, info
    
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.last_price = None
        return self._get_observation()

def make_env(merged_file, model_name, rank):
    """Create a single environment."""
    def _init():
        env = TradingEnv(merged_file, model_name)
        env.seed(42 + rank)
        return env
    return _init

def get_data_files():
    """Get all available data files for training and testing."""
    if not os.path.exists('merged'):
        raise FileNotFoundError("Merged data directory not found")
    
    # Get all merged files
    merged_files = []
    for fname in os.listdir('merged'):
        if fname.endswith('.csv'):
            # Parse stock symbol from filename (e.g., merged_AAPL_Aug-01-2024.csv)
            match = re.match(r'merged_([A-Z]+)_', fname)
            if match:
                symbol = match.group(1)
                merged_files.append((symbol, os.path.join('merged', fname)))
    
    if not merged_files:
        raise FileNotFoundError("No merged data files found")
    
    # Sort files by date (newest first)
    merged_files.sort(key=lambda x: x[1], reverse=True)
    
    # Use 80% for training, 20% for testing
    split_idx = int(len(merged_files) * 0.8)
    train_files = merged_files[:split_idx]
    test_files = merged_files[split_idx:]
    
    logger.info(f"Found {len(merged_files)} total data files")
    logger.info(f"Using {len(train_files)} files for training")
    logger.info(f"Using {len(test_files)} files for testing")
    
    return train_files, test_files

def train_agent(train_files, model_name="finbert", total_timesteps=10000, time_tracker=None):
    """Train the PPO agent on all training files."""
    
    # Check if we're on macOS
    is_macos = platform.system() == 'Darwin'
    
    if is_macos:
        # On macOS, use DummyVecEnv with fewer environments
        logger.info("Running on macOS - using DummyVecEnv with reduced parallelism")
        n_envs = min(4, len(train_files))  # Limit to 4 environments on macOS
        env = DummyVecEnv([
            make_env(merged_file, model_name, i)
            for i, (_, merged_file) in enumerate(train_files[:n_envs])
        ])
    else:
        # On other platforms, use SubprocVecEnv with full parallelism
        n_cpu = multiprocessing.cpu_count()
        n_envs = min(n_cpu, len(train_files))
        logger.info(f"Using {n_envs} parallel environments")
        
        # Set start method for multiprocessing
        if platform.system() != 'Windows':
            multiprocessing.set_start_method('spawn', force=True)
        
        env = SubprocVecEnv([
            make_env(merged_file, model_name, i)
            for i, (_, merged_file) in enumerate(train_files[:n_envs])
        ])
    
    # Calculate steps per update to ensure exact total timesteps
    n_steps = 1024  # Reduced from 2048 for better control
    batch_size = 64
    n_epochs = 10
    
    # Initialize PPO agent with optimized hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        device="cpu",  # Force CPU for PPO
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        )
    )
    
    # Train agent with progress tracking
    logger.info(f"Starting training for {total_timesteps} timesteps using {model_name}...")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    if time_tracker:
        time_tracker.checkpoint("Training")
    
    return model

def test_agent(model, test_files, model_name, time_tracker=None):
    """Test the trained agent on test files."""
    results = []
    
    for _, merged_file in test_files:
        try:
            # Create test environment
            env = TradingEnv(merged_file, model_name)
            
            # Run episode
            obs = env.reset()
            done = False
            episode_rewards = []
            portfolio_values = []
            initial_balance = env.initial_balance
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_rewards.append(reward)
                portfolio_value = info['balance'] + (info['position'] * info['current_price'])
                portfolio_values.append(portfolio_value)
            
            # Calculate metrics
            total_reward = sum(episode_rewards)
            final_value = portfolio_values[-1]
            return_pct = ((final_value - initial_balance) / initial_balance) * 100
            
            # Calculate max drawdown
            portfolio_values = np.array(portfolio_values)
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = np.min(drawdown) * 100 if len(drawdown) > 0 else 0
            
            # Calculate Sharpe ratio
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = np.sqrt(252) * (np.mean(returns) / np.std(returns)) if len(returns) > 0 and np.std(returns) > 0 else 0
            
            # Extract stock symbol from filename
            stock = os.path.basename(merged_file).split('_')[1]
            
            results.append({
                'stock': stock,
                'total_reward': total_reward,
                'final_value': final_value,
                'return_pct': return_pct,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            })
            
            logger.info(f"Test results for {stock} using {model_name}:")
            logger.info(f"  Final Value: ${final_value:.2f}")
            logger.info(f"  Return: {return_pct:.2f}%")
            logger.info(f"  Max Drawdown: {max_drawdown:.2f}%")
            logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
            
        except Exception as e:
            logger.error(f"Error testing {merged_file}: {str(e)}")
            continue
    
    if time_tracker:
        time_tracker.checkpoint("Testing")
    
    return pd.DataFrame(results)

def main():
    """Main function to train and evaluate the trading agent with different transformer models."""
    
    # List of transformer models to try
    transformer_models = ["finbert", "longformer", "longt5"]
    
    for model_name in transformer_models:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training with {model_name.upper()} model")
        logger.info(f"{'='*50}")
        
        # Initialize time tracker
        time_tracker = TimeTracker()
        time_tracker.start()
        
        try:
            # Get data files
            logger.info("Loading data files...")
            train_files, test_files = get_data_files()
            logger.info(f"Found {len(train_files)} training files and {len(test_files)} test files")
            time_tracker.checkpoint("Data Loading")
            
            # Train agent
            logger.info("Training agent...")
            model = train_agent(train_files, model_name=model_name, total_timesteps=1000, time_tracker=time_tracker)
            
            # Save model
            model_save_path = f"trading_agent_ppo_{model_name}"
            model.save(model_save_path)
            logger.info(f"Model saved successfully to {model_save_path}")
            time_tracker.checkpoint("Model Saving")
            
            # Test agent
            logger.info("Testing agent...")
            test_results = test_agent(model, test_files, model_name, time_tracker=time_tracker)
            
            # Print test results
            logger.info("\nTest Results:")
            logger.info(f"Average Return: {test_results['return_pct'].mean():.2f}%")
            logger.info(f"Average Sharpe Ratio: {test_results['sharpe_ratio'].mean():.2f}")
            logger.info(f"Average Max Drawdown: {test_results['max_drawdown'].mean():.2f}%")
            
            # Save test results
            results_file = f"test_results_{model_name}.csv"
            test_results.to_csv(results_file, index=False)
            logger.info(f"Test results saved to {results_file}")
            
            # Print timing summary
            time_tracker.print_summary()
            
        except Exception as e:
            logger.error(f"Error during training/testing with {model_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 