"""
Script to run a single test case from Task 2.
This is useful for running individual experiments without running all test cases.

Usage:
    python run_single_test.py --test_case 2 --epsilon 0.01 --ic_type sin --epochs 5000
"""

import argparse
from train_allen_cahn import run_test_case


def main():
    parser = argparse.ArgumentParser(description='Run a single Allen-Cahn test case')
    parser.add_argument('--test_case', type=int, default=2, 
                       choices=[2, 3, 4],
                       help='Test case number (2, 3, or 4)')
    parser.add_argument('--epsilon', type=float, default=0.01,
                       help='Diffusivity parameter ε')
    parser.add_argument('--ic_type', type=str, default='sin',
                       choices=['sin', 'step'],
                       help='Initial condition type: sin or step')
    parser.add_argument('--epochs', type=int, default=10000,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    print(f"\nRunning Test Case {args.test_case}")
    print(f"  ε = {args.epsilon}")
    print(f"  Initial Condition: {args.ic_type}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}\n")
    
    results = run_test_case(
        test_case=args.test_case,
        epsilon=args.epsilon,
        ic_type=args.ic_type,
        epochs=args.epochs,
        lr=args.lr
    )
    
    print("\nExperiment completed successfully!")
    return results


if __name__ == "__main__":
    main()

