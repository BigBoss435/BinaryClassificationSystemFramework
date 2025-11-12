"""
Quick launch script for melanoma detection agent
"""

from inference_agent import MelanomaAIAgent
import sys

def main():
    # Initialize agent
    agent = MelanomaAIAgent(checkpoint_path="checkpoints/best_model.pth")
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python launch_agent.py <image_path>")
        return
    
    # Process image
    print(f"\nAnalyzing: {image_path}...")
    result = agent.process_single_case(image_path, use_tta=True)
    
    # Display results
    print("\n" + "="*60)
    print("DIAGNOSIS REPORT")
    print("="*60)
    print(f"Classification: {result['diagnosis']['classification']}")
    print(f"Confidence: {result['diagnosis']['confidence_score']:.1%}")
    print(f"Risk Level: {result['diagnosis']['risk_level']}")
    print("\nRecommendations:")
    for rec in result['recommendations']['next_steps']:
        print(f"  • {rec}")
    print("="*60)
    
    # Save report
    agent.export_report(result, "latest_diagnosis.json")

if __name__ == "__main__":
    main()