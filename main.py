from src.dpo_training import main as train
from src.run_inference import main as infer

def main():
    train()
    infer()

if __name__ == "__main__":
    main()