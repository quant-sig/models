import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

from QuantConnect import Resolution
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Data.Market import TradeBar

from model import LSTMGCN

class StarPlatinum(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2014, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)

        # Add futures
        self.es_futures = self.AddFuture("ES", Resolution.Hour)
        self.vx_futures = self.AddFuture("VX", Resolution.Hour)
        
        # Store historical data
        self.history_data = []
        self.lookback = 20000
        self.hours_since_retrain = 0

        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMGCN(12).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        self.criterion = nn.L1Loss()
        self.l1_lambda = 1e-5
        
        # Initial training once we have enough data
        history = self.History(["ES", "VX"], self.lookback, Resolution.Hour)
        if not history.empty:
            # Process historical data
            self.process_history(history)
            # Initial training
            self.initial_training()

    def create_train_loader(self):
        # Convert stored data to tensors
        X = torch.tensor([d['features'] for d in self.history_data[-self.lookback:]])
        y = torch.tensor([d['targets'] for d in self.history_data[-self.lookback:]])
        
        # Create dataset
        dataset = TensorDataset(X, y)
        
        # Create dataloader
        batch_size = 32  # Or whatever batch size you want
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        return train_loader

    def initial_training(self):
        train_loader = self.create_train_loader()
        
        for epoch in range(120):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(x)
                
                loss = self.criterion(pred, y)
                l1_loss = 0
                for param in self.model.parameters():
                    l1_loss += torch.sum(torch.abs(param))
                loss += self.l1_lambda * l1_loss
                
                loss.backward()
                self.optimizer.step()
                
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                }, f'checkpoint_{epoch}.pt')
        
    def OnData(self, data):
        if not (self.es_futures.Current and self.vx_futures.Current):
            return

        # Get features
        features = self.get_features()
        if features is None:
            return
            
        # Make prediction
        with torch.no_grad():
            x = torch.tensor(features).float().to(self.device)
            pred = self.model(x.unsqueeze(0))
            
        # Store data for retraining
        targets = self.calculate_targets()
        self.history_data.append({
            'features': features,
            'targets': targets
        })
        
        # Maintain rolling window
        if len(self.history_data) > self.lookback:
            self.history_data.pop(0)
            
        # Check if retraining needed
        self.hours_since_retrain += 1
        if self.hours_since_retrain >= 1500:
            self.hours_since_retrain = 0
            self.retrain()
            
        # Trade based on predictions
        self.trade_logic(pred)

    def retrain(self):
        train_loader = self.create_train_loader()
        
        for epoch in range(25):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(x)
                
                loss = self.criterion(pred, y)
                l1_loss = 0
                for param in self.model.parameters():
                    l1_loss += torch.sum(torch.abs(param))
                loss += self.l1_lambda * l1_loss
                
                loss.backward()
                self.optimizer.step()

    def get_features(self):
        features = []
        # Get next 4 ES contracts
        es_contracts = [c for c in self.es_futures.Chain 
                       if c.Expiry > self.Time][:4]
        # Get next 8 VX contracts
        vx_contracts = [c for c in self.vx_futures.Chain 
                       if c.Expiry > self.Time][:8]
        
        if len(es_contracts) < 4 or len(vx_contracts) < 8:
            return None
            
        # ES contracts
        for contract in es_contracts:
            if contract.Price == 0:
                return None
            features.append(contract.Price)
            
        # VX contracts
        for contract in vx_contracts:
            if contract.Price == 0:
                return None
            features.append(contract.Price)
            
        return features

    def calculate_targets(self):
        returns = self.calculate_returns()
        volatility = self.calculate_volatility()
        volume = self.count_top_of_book_updates()
        return [returns, volatility, volume]

    def trade_logic(self, pred):
        """
        here is where you'd do the trading based on the predictions
        apply strategies, etc, whatever.
        """
        predicted_returns = pred[0].item()
        predicted_vol = pred[1].item()
        predicted_volume = pred[2].item()
        
        # Example simple logic
        if predicted_returns > some_threshold:
            self.SetHoldings(self.es_futures.Contracts[0].Symbol, 1)
