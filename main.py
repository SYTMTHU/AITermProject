import argparse
import torch
from classifier import *
import time
import math
import pickle
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--datapath', type=str, default='./',
                    help='location of the data corpus')
parser.add_argument('--nhid1', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nhid2', type=int, default=256,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1000, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='report interval')
args = parser.parse_args()

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

nentities = 26
nout = 5

path = args.datapath

f_target_test = open(path+'target_test', 'rb')

f_target_valid = open(path+'target_valid', 'rb')
f_target_train = open(path+'target_train', 'rb')

f_input_test = open(path+'input_test', 'rb')
f_input_valid = open(path+'input_valid', 'rb')
f_input_train = open(path+'input_train', 'rb')

train_input = pickle.load(f_input_train)
test_input = pickle.load(f_input_valid)
valid_input = pickle.load(f_input_test)

train_output = pickle.load(f_target_train)
test_output = pickle.load(f_target_test)
valid_output = pickle.load(f_target_valid)


lr = args.lr
batch_size = args.batch_size
model = Classifier(nentities, nout)

if torch.cuda.is_available():
    model = model.cuda()
    Tensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    FloatTensor = torch.FloatTensor


def train(model, nentities, nout, nepoch, input, output, batch_size):
    criterion = torch.nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    best_val_accuracy = None

    try:
        for epoch in range(1, args.epochs + 1):
            correct = 0
            epoch_start_time = time.time()
            total_loss = 0
            start_time = time.time()
            count = 0
            while True:
                flag = True
                if count + batch_size <= len(input):
                    batch_in = FloatTensor(input[count:count + batch_size])  # (batch_size, nentities)
                    batch_out = LongTensor(output[count:count + batch_size])  # (batch_size, nout)
                else:
                    batch_in = FloatTensor(input[count:])  # (batch_size, nentities)
                    batch_out = LongTensor(output[count:])  # (batch_size, nout)
                    flag = False
                real_out = model(Variable(batch_in))
                loss = criterion(real_out, Variable(batch_out))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.data
              #  print(real_out)
                max_indexes = np.argmax(real_out.data, axis=1)
              #  print(type(max_indexes))

                correct = (max_indexes == batch_out).sum()
                accuracy_rate = correct/len(real_out)
                cur_loss = total_loss[0] / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | accuracy {:5.2f}'.format(
                    epoch, int(count/batch_size), len(input) // batch_size, lr,
                                  elapsed * 1000 / args.log_interval, cur_loss, accuracy_rate))
                total_loss = 0
                start_time = time.time()
                count += batch_size
                if not flag:
                    break
            val_accuracy = evaluate(model, valid_input, valid_output, batch_size) #TODO: finish the validation data
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid accuracy: {:5.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_accuracy))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_accuracy or val_accuracy < best_val_accuracy:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_accuracy = val_accuracy
          #  else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                #lr /= 4.0

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    return model


def evaluate(model, input, output, batch_size):
  #  total_accuracy = 0
   # criterion = torch.nn.CrossEntropyLoss()
  #  optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4)
    count = 0
    correct = 0
 #   start_time = time.time()
    while True:
        flag = True
        if count + batch_size <= len(input):
            batch_in = FloatTensor(input[count:count + batch_size])  # (batch_size, nentities)
            batch_out = LongTensor(output[count:count + batch_size])  # (batch_size, nout)
        else:
            batch_in = FloatTensor(input[count:])  # (batch_size, nentities)
            batch_out = LongTensor(output[count:])  # (batch_size, nout)
            flag = False
        real_out = model(Variable(batch_in))
       # loss = criterion(real_out, batch_out)
      #  optimizer.zero_grad()
      #  loss.backward()
      #  optimizer.step()
       # total_loss += loss.data
       # cur_loss = total_loss[0] / args.log_interval
       # elapsed = time.time() - start_time
       # total_loss = 0
       # start_time = time.time()
        count += batch_size
        max_indexes = np.argmax(real_out.data, axis=1)
      #  print(type(max_indexes))
      #  print(type(batch_out))
        correct += (max_indexes == batch_out).sum()
        if not flag:
            break
    return correct/len(input)

# Run on test data.
if __name__ == '__main__':
    final_model = train(model, nentities, nout, args.epochs, train_input, train_output, batch_size)
    test_accuracy = evaluate(final_model, test_input, test_output, batch_size)
    print('=' * 89)
    print('| End of training | test accuracy {:5.2f}'.format(
        test_accuracy))
