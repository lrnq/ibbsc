import argparse
import re

def default_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation","-a", dest="activation", default="tanh",
                        help="Possible activation functions tanh, relu, relu6, elu")

    parser.add_argument("--batch_size","-bs", dest="batch_size", default="256",
                         help="Size of the batches")

    parser.add_argument("--learning_rate","-lr", dest="learning_rate", default=0.0004,
                        type=float, help="The learning rate of the network")

    parser.add_argument("--data","-d", dest="data", default="../data/var_u.mat",
                        help="Path to the datafile")
        
    parser.add_argument("--save_path","-sp", dest="save_path", default="../data/saved_data",
                        help="Path to the folder for saving data")

    parser.add_argument("--epochs", "-e", dest="epochs", default=8000,
                        type=int, help="Number of epochs to run")

    parser.add_argument("--num_runs","-num", dest="num_runs", default=40,
                        type=int, help="Number of times to run the network")  

    parser.add_argument("--mi_methods", "-mi", dest="mi_methods", default="[adaptive]",
                        help="Number of times to run the network. Can contain multiple methods")  

    parser.add_argument("--try_gpu","-g", dest="try_gpu", default=0,
                        type=int, help="Whether or not to try and run on the GPU. Seeds may not be set here. Expects a 0 or 1.")

    parser.add_argument("--num_bins", "-nb", dest="num_bins", default="[30]",
                        help="Number of bins to use for MI estimation. Expects list of values.") 

    parser.add_argument("--layer_sizes", "-ls", dest="layer_sizes", default="[12, 10, 7, 5, 4, 3, 2]",
                        help="Number of times to run the network")   
                    
    args = parser.parse_args()

    args.mi_methods = [x.strip() for x in re.findall('\[(.*?)\]', args.mi_methods)[0].split(",")]   
    args.num_bins = [int(x.strip()) for x in re.findall('\[(.*?)\]', args.num_bins)[0].split(",")]
    args.layer_sizes = [int(x.strip()) for x in re.findall('\[(.*?)\]', args.layer_sizes)[0].split(",")]
    args.try_gpu = bool(args.try_gpu)
    if args.batch_size != "full":
        args.batch_size = int(args.batch_size)

    return args

if __name__ == "__main__":
    args = default_params()