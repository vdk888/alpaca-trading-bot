from flask import Flask, render_template, jsonify
import threading
import os
import json
from datetime import datetime
import logging
import time
from queue import Queue
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# In-memory storage for data
class DataStore:
    def __init__(self):
        self.data = {}
        self.trading_signals = []
        self.portfolio_history = []
        self.last_update = {}
        self.last_prices = {}
        self.active_symbols = set()
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

    def update_price_data(self, symbol, data_dict):
        """Update price data for a symbol"""
        self.last_update[symbol] = datetime.now().isoformat()
        self.data[symbol] = data_dict
        self.last_prices[symbol] = data_dict.get('current_price', None)
        self.active_symbols.add(symbol)
        
        # Save data to disk for persistence
        with open(f'data/{symbol}_latest.json', 'w') as f:
            json.dump(data_dict, f)
    
    def add_trading_signal(self, signal_data):
        """Add a new trading signal to the history"""
        signal_data['timestamp'] = datetime.now().isoformat()
        self.trading_signals.append(signal_data)
        if len(self.trading_signals) > 100:  # Keep only the last 100 signals
            self.trading_signals = self.trading_signals[-100:]
            
        # Save signals to disk
        with open('data/trading_signals.json', 'w') as f:
            json.dump(self.trading_signals, f)
    
    def update_portfolio(self, portfolio_data):
        """Update portfolio history"""
        portfolio_data['timestamp'] = datetime.now().isoformat()
        self.portfolio_history.append(portfolio_data)
        if len(self.portfolio_history) > 1000:  # Keep a reasonable history
            self.portfolio_history = self.portfolio_history[-1000:]
            
        # Save portfolio data to disk
        with open('data/portfolio_history.json', 'w') as f:
            json.dump(self.portfolio_history, f)
    
    def get_symbol_data(self, symbol):
        """Get data for a specific symbol"""
        return self.data.get(symbol, {})
    
    def get_latest_prices(self):
        """Get latest prices for all symbols"""
        return self.last_prices
    
    def get_active_symbols(self):
        """Get list of active symbols"""
        return list(self.active_symbols)
    
    def load_from_disk(self):
        """Load persistent data from disk on startup"""
        try:
            # Load trading signals
            if os.path.exists('data/trading_signals.json'):
                with open('data/trading_signals.json', 'r') as f:
                    self.trading_signals = json.load(f)
            
            # Load portfolio history
            if os.path.exists('data/portfolio_history.json'):
                with open('data/portfolio_history.json', 'r') as f:
                    self.portfolio_history = json.load(f)
            
            # Load latest symbol data
            for filename in os.listdir('data'):
                if filename.endswith('_latest.json'):
                    symbol = filename.split('_')[0]
                    with open(f'data/{filename}', 'r') as f:
                        self.data[symbol] = json.load(f)
                    self.active_symbols.add(symbol)
                    
                    if 'current_price' in self.data[symbol]:
                        self.last_prices[symbol] = self.data[symbol]['current_price']
        except Exception as e:
            logging.error(f"Error loading data from disk: {e}")

# Create data store instance
data_store = DataStore()
data_store.load_from_disk()

@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('dashboard.html', 
                          active_symbols=data_store.get_active_symbols(),
                          last_update=data_store.last_update)

@app.route('/api/symbols')
def get_symbols():
    """Return list of active symbols"""
    return jsonify({
        'symbols': data_store.get_active_symbols(),
        'last_update': data_store.last_update
    })

@app.route('/api/data/<symbol>')
def get_symbol_data(symbol):
    """Return data for a specific symbol"""
    return jsonify(data_store.get_symbol_data(symbol))

@app.route('/api/prices')
def get_prices():
    """Return latest prices for all symbols"""
    return jsonify(data_store.get_latest_prices())

@app.route('/api/signals')
def get_signals():
    """Return recent trading signals"""
    return jsonify(data_store.trading_signals)

@app.route('/api/portfolio')
def get_portfolio():
    """Return portfolio history"""
    return jsonify(data_store.portfolio_history)

@app.route('/health')
def health_check():
    """Health check endpoint for Replit"""
    return jsonify({"status": "ok", "message": "Trading bot dashboard is running"})

def run_flask_server():
    """Run the Flask server in a separate thread"""
    # Check if running on Replit
    if os.getenv('REPL_ID') and os.getenv('REPL_SLUG'):
        # Running on Replit - use 0.0.0.0 and port 8080
        print("Running on Replit - using port 8080")
        app.run(host='0.0.0.0', port=8080)
    else:
        # Running locally - use port 8081
        print("Running locally - using port 8081")
        app.run(host='0.0.0.0', port=8081)

# This allows importing the datastore from the main script
def get_data_store():
    return data_store