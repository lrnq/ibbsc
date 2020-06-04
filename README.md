# ibbsc

## Setup
It is required to [Conda](https://docs.conda.io/en/latest/) on your system to use the provided `YAML` file from the repo. When conda is install you can create an enviroment with the requirements by running 
`conda env create -f ibbsc.yml`

## Usage 
The section below assume that `python` is linked to the correct version (it will be inside the Conda enviroment).
```  
Usage:  
    cd ibbsc
    python main.py [Parameters]
  
Parameters:  
    -h    --help                        Prints help similar to this.
    -a    --activation                  Sets the activation function for the hidden layers, but the last.  
    -bs   --batch_size                  Batch size for training.   
    -d    --data                        Path to the data file used for training.
    -lr   --learning_rate               Learning rate for Adam optimizer.
    -sp   --save_path                   Path to the folder for saving data.
    -e    --epochs                      Number of epochs to train on per run.  
    -num  --num_runs                    Number of times to run the network.
    -mi   --mi_methods                  List of method(s) for estimating the mutual information.
    -g    --try_gpu                     Try to train on a GPU. Note that CUDA seeds might not have been set.
    -nb   --num_bins                    Number of bins to use for discretization. 
    -ls   --layer_sizes                 List of the layers sizes of the network. 
    -pr   --plot_results                Plot the information plane of the data just generated.
    -sm   --save_max_vals               Save max values for each layer at each epoch.
    -ste  --save_train_error            Save training error as a function of the epochs for each run.
    -smi  --save_mutual_information     Save mutual information after each epoch.
    -sf   --start_from                  Start running experiment from a specific run number.
    -ts   --test_size                   Size of the test data.
```

For the default parameter values see `ibbsc/default_params.py`

## Examples

**Running these exampes takes roughly 15-25 hours, dependend on the example, on a decent laptop.** 

To generate the information plane averaged over 40 runs for a network with layer sizes `[12,10,7,5,4,3,2]` (default network) the `tanh` activation function and the mutual information estimated using:  

1) The adaptive binning strategy:  
```
cd ibbsc  
python main.py -a=tanh -bs=256 -lr=0.0004 -sp="../data/saved_data" -num=40 -mi="[adaptive]" -nb="[30]" 
```

2) The adaptive binning and the fixed binning strategy:  
```
cd ibbsc  
python main.py -a=tanh -bs=256 -lr=0.0004 -sp="../data/saved_data" -num=40 -mi="[fixed,adaptive]" -nb="[30]" 
```

3) The adaptive binning and the fixed binning strategy for multiple number of bins:  
```
cd ibbsc  
python main.py -a=tanh -bs=256 -lr=0.0004 -sp="../data/saved_data" -num=40 -mi="[fixed,adaptive]" -nb="[30,100]" 
```
