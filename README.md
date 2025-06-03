# Code for "Scalable quantum dynamics compilation via quantum machine learning"

Here, we provide Python code to reproduce the results of [https://arxiv.org/abs/2409.16346](https://arxiv.org/abs/2409.16346).
## Instructions 
Clone the repo and create a fresh Conda environment with Python 3.10 and use Pip to install `requirements.txt`. 
Alternatively, you can run `sh setup.sh`, which will create the conda environemnt for you and activate it. 

The main driver codes are `run_1d.py` and `run_evolution.py` in the `vff` folder.

If you want to run your own scripts, look at `reproduce.py` for examples.

## GPU support
If you have a GPU available, PyTorch and Quimb should detect the GPU automatically and run the code on GPU.

## Reproducing the figures

To reproduce the figures in the paper, activate the Conda environment described above and run `python reproduce.py`.
The data needed to produce all the plots has been added to the repo in the `data` folder.
Figures will automatically be saved in `figures`

If you delete the folder `data`, then `reproduce.py` will generate all the data
for you, but this will likely take a long time and require HPC resources.

As of September 18th 2024, the following works on MacOS 14.6.1 and Conda 24.3.0:

```bash
sh setup.sh
conda activate var_comp
python -u reproduce.py
```

## Help

Please reach out on Github or via email if you have any questions about the code.
