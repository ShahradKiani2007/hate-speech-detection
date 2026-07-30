from run_train import main

# alias so `python run.py` and `python run_train.py` both run the full training
# pipeline (data prep -> split -> augment -> features -> classic + lstm + bert).
if __name__ == "__main__":
    main()
