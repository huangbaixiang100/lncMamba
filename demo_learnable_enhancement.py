#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for the learnable enhancement factor.
Shows how to monitor and use the learnable attention enhancement factor.
"""

import torch
import numpy as np
from nnLayer import Improved_DeepPseudoLabelwiseAttention

def demo_learnable_enhancement_factor():
    """Demonstrate the functionality of the learnable enhancement factor."""
    
    print("=== Learnable Enhancement Factor Demo ===\n")
    
    # Create a simple tokenizer mock
    class MockTokenizer:
        def __init__(self):
            self.tkn2id = {'TTT': 1, '[PAD]': 0, 'AAA': 2}
    
    tokenizer = MockTokenizer()
    
    # Create attention modules with different initial enhancement factors
    print("1. Create attention modules with different initial enhancement factors:")
    
    attention_modules = []
    initial_factors = [1.0, 1.5, 2.0, 2.5]
    
    for factor in initial_factors:
        module = Improved_DeepPseudoLabelwiseAttention(
            inSize=64, 
            classNum=10, 
            L=-1, 
            tokenizer=tokenizer,
            sequences=["TTT"], 
            enhanceFactor=factor
        )
        attention_modules.append(module)
        print(f"   Initial factor {factor} -> Actual enhancement factor: {module.get_enhancement_factor():.3f}")
        print(f"   Raw parameter value: {module.get_raw_enhancement_parameter():.3f}")
    
    print("\n2. Simulate parameter updates during training:")
    
    # Choose one module for demo
    demo_module = attention_modules[1]  # the module with initial factor 1.5
    
    # Create dummy input data
    batch_size, seq_len, embed_dim = 2, 10, 64
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Optimizer
    optimizer = torch.optim.Adam(demo_module.parameters(), lr=0.01)
    
    print("   Enhancement factor before training:")
    print(f"   Actual enhancement factor: {demo_module.get_enhancement_factor():.3f}")
    print(f"   Raw parameter value: {demo_module.get_raw_enhancement_parameter():.3f}")
    
    # Simulate several training steps
    for step in range(5):
        optimizer.zero_grad()
        
        # Forward
        output, attention_weights = demo_module(x)
        
        # Dummy loss (for demo only)
        loss = torch.sum(output**2)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        print(f"   Step {step+1}: enhancement = {demo_module.get_enhancement_factor():.3f}, "
              f"raw = {demo_module.get_raw_enhancement_parameter():.3f}, "
              f"loss = {loss.item():.3f}")
    
    print("\n3. Constraint effect of the enhancement factor:")
    print("   Because we use sigmoid(x) * 2 + 1 transformation:")
    print("   - The enhancement factor is constrained within [1.0, 3.0]")
    print("   - This ensures reasonable enhancement")
    print("   - The model can learn an appropriate enhancement strength")
    
    # Test extreme raw parameter values
    test_raw_values = [-10, -1, 0, 1, 10]
    print(f"\n   Raw parameter -> actual enhancement mapping:")
    for raw_val in test_raw_values:
        enhanced_factor = torch.sigmoid(torch.tensor(raw_val)).item() * 2.0 + 1.0
        print(f"   {raw_val:4} -> {enhanced_factor:.3f}")
    
    print("\n4. Practical tips for training:")
    print("   - A reasonable initial enhancement factor is 1.5")
    print("   - Monitor the trend of the enhancement factor during training")
    print("   - If it trends toward 1.0, the model considers enhancement unnecessary")
    print("   - If it trends toward 3.0, stronger enhancement is needed")
    print("   - You may adjust the sigmoid mapping range to fit different needs")
    
    print("\n=== Demo finished ===")

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    demo_learnable_enhancement_factor() 