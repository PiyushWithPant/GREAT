`main.py` for Data Poisoning


```
# Clean baseline
python main.py --task clean --model baseline_0 --trigger clean

# SUDO baseline
python main.py --task sudo --model baseline_A

# K=2000 baseline
python main.py --task k2000 --model MODEL --trigger k_2000

# Print example prompts from evaluation datasets
python main.py --task print_examples --model baseline_0 --trigger clean

```


---


`subpopulation_selection_using_zs_classifier.py` for getting the desired subpopulation from the dataset