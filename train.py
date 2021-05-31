import argparse
import functions 

parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('data_dir' , action='store' , default="./flowers/")
parser.add_argument('--gpu' , dest='gpu', action='store', default='gpu')
parser.add_argument('--epochs' , dest='epochs' , action ='store' , type = int, default= 15)
parser.add_argument('--hideln_layer1' , dest='hidden_layer1' , action ='store' , type = int, default= 500)
parser.add_argument('--hideln_layer2' , dest='hidden_layer2' , action ='store' , type = int, default= 250)
parser.add_argument('--structure' , dest='structure' , action ='store' , type = str, default= 'densenet121')
parser.add_argument('--lr' , dest='lr' , action ='store' , type = int, default= 0.001)
parser.add_argument('--dropout' , dest='dropout' , action ='store' , type = int, default= 0.05)                    
# parser.add_argument('--lr' , dest='lr' , action ='store' , type = int, default= 0.001)
parser.add_argument('--save_directory' , dest='save_directory' , action ='store' , default = './checkpoint.pth')


parser = parser.parse_args()
epochs = parser.epochs
lr = parser.lr
structure = parser.structure
dropout = parser.dropout
hidden_layer1 = parser.hidden_layer1
hidden_layer2 = parser.hidden_layer2
power = parser.gpu



train_loaders , valid_loaders , test_loaders = functions.read_data(data_dir)
## load the model
model, optimizer , criterion = functions.model_setup (structure ,epochs,dropout, hidden_layer1, hidden_layer2, lr, power)
## train the model
functions.train (model , epochs , criterion , optimizer, train_loaders , valid_loaders, power)
## save the model
functions.save_checkpoint (path, hidden_layer1, hidden_layer2,dropout,lr,epochs)

