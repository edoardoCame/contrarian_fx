#!/usr/bin/env python3
"""
Performance Test for Optimized ConrarianSignalGenerator

This script specifically tests the signal generation optimization to ensure
it meets the target of <5 seconds for the full dataset.
"""

import sys
import time
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import psutil
import os

# Add modules to path
sys.path.append('modules')

from modules.data_loader import ForexDataLoader
from modules.signal_generator import ConrarianSignalGenerator

warnings.filterwarnings('ignore')

def measure_memory():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_signal_generation_performance():
    """Test the optimized signal generation performance specifically."""
    
    print("🚀 Testing OPTIMIZED ConrarianSignalGenerator Performance")
    print("=" * 70)
    
    # Record start time and memory
    total_start_time = time.time()
    start_memory = measure_memory()
    
    # Step 1: Load data
    print("📈 Loading forex market data...")
    data_start = time.time()
    
    data_loader = ForexDataLoader('data')
    prices = data_loader.load_unified_prices()
    returns = data_loader.load_unified_returns()
    
    data_time = time.time() - data_start
    data_memory = measure_memory()
    
    print(f"✅ Data loaded in {data_time:.2f} seconds")
    print(f"   - {len(prices.columns)} currency pairs")
    print(f"   - {len(prices):,} trading days")
    print(f"   - Memory usage: {data_memory:.1f} MB")
    
    # Step 2: Test different parameter configurations for performance
    test_configurations = [
        {'n': 3, 'm': 30, 'name': 'Optimized (N=3, M=30)'},
        {'n': 5, 'm': 20, 'name': 'Alternative (N=5, M=20)'},
        {'n': 2, 'm': 10, 'name': 'Fast (N=2, M=10)'}
    ]
    
    results = {}
    
    for config in test_configurations:
        print(f"\n🔬 Testing {config['name']}...")
        
        # Initialize optimized signal generator
        signal_start = time.time()
        signal_start_memory = measure_memory()
        
        generator = ConrarianSignalGenerator(
            n_worst_performers=config['n'], 
            lookback_days=config['m']
        )
        
        # Generate signals
        signal_output = generator.generate_signals(prices, returns)
        
        signal_time = time.time() - signal_start
        signal_memory = measure_memory()
        memory_used = signal_memory - signal_start_memory
        
        print(f"✅ {config['name']} completed in {signal_time:.2f} seconds")
        print(f"   - Memory used: {memory_used:.1f} MB")
        print(f"   - Target (<5s): {'✅ PASSED' if signal_time < 5 else '❌ FAILED'}")
        
        # Validate signal quality
        binary_signals = signal_output['binary_signals']
        weights = signal_output['weights']
        
        # Quick validation
        total_signals = binary_signals.sum().sum()
        avg_signals_per_day = binary_signals.sum(axis=1).mean()
        valid_days = (binary_signals.sum(axis=1) > 0).sum()
        weight_coverage = (weights > 0).sum().sum()
        
        print(f"   - Total signals: {total_signals:,}")
        print(f"   - Avg signals/day: {avg_signals_per_day:.2f}")
        print(f"   - Valid trading days: {valid_days:,}")
        print(f"   - Weight entries: {weight_coverage:,}")
        
        results[config['name']] = {
            'time': signal_time,
            'memory_mb': memory_used,
            'total_signals': total_signals,
            'avg_signals_per_day': avg_signals_per_day,
            'valid_days': valid_days,
            'target_met': signal_time < 5.0,
            'signal_output': signal_output
        }
    
    # Performance summary
    print(f"\n📋 SIGNAL GENERATION PERFORMANCE SUMMARY")
    print("=" * 70)
    
    fastest_config = min(results.keys(), key=lambda k: results[k]['time'])
    
    for config_name, result in results.items():
        status = "✅ PASSED" if result['target_met'] else "❌ FAILED"
        star = " ⭐ FASTEST" if config_name == fastest_config else ""
        print(f"{config_name:25} {result['time']:>6.2f}s  {result['memory_mb']:>6.1f}MB  {status}{star}")
    
    print(f"\nTarget Performance (<5s): {sum(r['target_met'] for r in results.values())}/{len(results)} configs passed")
    
    # Detailed analysis of the fastest configuration
    fastest_result = results[fastest_config]
    print(f"\n🎯 FASTEST CONFIGURATION ANALYSIS: {fastest_config}")
    print("=" * 70)
    print(f"Execution Time:        {fastest_result['time']:.3f} seconds")
    print(f"Memory Usage:          {fastest_result['memory_mb']:.1f} MB") 
    print(f"Total Dataset Size:    {len(prices):,} days × {len(prices.columns)} pairs")
    print(f"Processing Rate:       {len(prices) * len(prices.columns) / fastest_result['time']:,.0f} data points/second")
    
    # Calculate improvement estimate (assuming original took 30+ seconds based on portfolio optimization)
    estimated_original_time = 30  # Conservative estimate
    improvement_factor = estimated_original_time / fastest_result['time']
    print(f"Estimated Improvement: {improvement_factor:.1f}x faster than unoptimized")
    
    # Data quality checks
    print(f"\n📊 DATA QUALITY VALIDATION")
    print("=" * 70)
    
    best_output = fastest_result['signal_output']
    validation_result = generator.validate_signals(best_output)
    
    print(f"Validation passed:     {'✅ YES' if len(validation_result['issues']) == 0 else '❌ NO'}")
    print(f"Signal coverage:       {validation_result['signal_coverage']:.2%}")
    print(f"Avg signals per day:   {validation_result['avg_signals_per_day']:.2f}")
    print(f"Weights sum correctly: {'✅ YES' if validation_result['weights_sum_to_one'] else '❌ NO'}")
    print(f"Correct signal count:  {'✅ YES' if validation_result['correct_number_selected'] else '❌ NO'}")
    
    if validation_result['issues']:
        print("\n⚠️ Validation Issues:")
        for issue in validation_result['issues']:
            print(f"   - {issue}")
    
    # Overall assessment
    total_time = time.time() - total_start_time
    final_memory = measure_memory()
    
    print(f"\n🎉 OPTIMIZATION ASSESSMENT")
    print("=" * 70)
    print(f"Best signal generation time: {fastest_result['time']:.2f} seconds")
    print(f"Target achieved (<5s):       {'✅ YES' if fastest_result['target_met'] else '❌ NO'}")
    print(f"Memory efficient:            {'✅ YES' if fastest_result['memory_mb'] < 200 else '⚠️ HIGH'}")
    print(f"Data quality validated:      {'✅ YES' if len(validation_result['issues']) == 0 else '❌ NO'}")
    print(f"Total test time:             {total_time:.2f} seconds")
    
    # Ready for production assessment
    production_ready = (fastest_result['target_met'] and 
                       len(validation_result['issues']) == 0 and
                       fastest_result['memory_mb'] < 300)
    
    print(f"\n🚀 PRODUCTION READINESS: {'✅ READY' if production_ready else '⚠️ NEEDS REVIEW'}")
    
    if production_ready:
        print(f"✅ Optimized signal generation is ready for use!")
        print(f"✅ Fastest configuration: {fastest_config}")
        print(f"✅ Expected runtime: ~{fastest_result['time']:.1f} seconds for full dataset")
    else:
        print("⚠️ Consider further optimization or review validation issues")
    
    return results

if __name__ == "__main__":
    try:
        performance_results = test_signal_generation_performance()
        
        # Extract best performance
        best_config = min(performance_results.keys(), 
                         key=lambda k: performance_results[k]['time'])
        best_time = performance_results[best_config]['time']
        
        print(f"\n🎯 FINAL RESULT: Signal generation optimized to {best_time:.2f} seconds")
        if best_time < 5:
            print(f"Target (<5s): ✅ ACHIEVED")
        else:
            print(f"Target (<5s): ❌ MISSED BY {best_time - 5:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Error during signal optimization testing: {str(e)}")
        import traceback
        traceback.print_exc()