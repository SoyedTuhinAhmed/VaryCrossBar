import argparse
from torchvision import datasets, transforms
from model import *
from stuck_at_faults import *


def load_data(transforms, dataset, batch_size, test_kwargs):
    if dataset == 'mist':
        testset = datasets.MNIST('../data', train=False, transform=transforms)
        test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)


def test(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--path', default=False,
                        help='path for pretrained model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        test_kwargs.update(cuda_kwargs)
    load_data(transforms, 'mnist', args.batch_size, device)
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()

    simulate_SA_faults(model, args.path, criterion)


def simulate_SA_faults(model, path, N, criterion):
    """ Load the model"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Best acc ', checkpoint['test_accuracy'])  # assumes model saves checkpoint accuracy

    print("Evaluating Stuck-At Faults:---------------------------")
    test_runs, best_acc = N, 0  # define test runs
    fi = StuckAtFaults()  # define fault injection class
    start_percent = 5
    end_percent = 20

    for percent in range(start_percent, end_percent, 5):
        print('Percentage of SA both faults: ', percent, '\n')
        test_accs = []
        for test_run in range(test_runs):
            model = fi.FI_SA_Both(percent, model, first=True)

            bn_stat_calibrate(model)

            test_acc = test(model, criterion)
            test_accs.append(test_acc)
            if best_acc < test_acc:
                print('---------------------- best acc updated--------------------')
                best_acc = test_acc
            print("Epochs {} Test accuracy: {:.4f} Best accuracy: {:.4f} -----------".format(test_run, test_acc,
                                                                                             best_acc))
            model.load_state_dict(checkpoint['model_state_dict'])

        mean_acc = np.round(np.mean(test_accs), 3)
        std_acc = np.round(np.std(test_accs), 3)
        print('percent ', percent, ' mean acc ', mean_acc, ' deviation ', std_acc)


if __name__ == '__main__':
    """
    global variables
    """
    main()
