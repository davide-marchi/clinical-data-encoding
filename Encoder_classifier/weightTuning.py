from classifier import ClassifierBinary
from modelEncoderDecoderAdvancedV2 import IMEO
from torch.nn import Sequential
from torch.nn.functional import binary_cross_entropy
from torcheval.metrics.functional import binary_accuracy
from torch.utils.data import DataLoader, TensorDataset
import torch


def tune_jointly(imeo:IMEO, classifier:ClassifierBinary, 
         tr, tr_o, vl, vl_o, 
         lr:float, ep:int, wd:float,
         batch_size:int, patience:int, 
         print_time:int, device)->"dict[str:'list[float]']":
    '''
    imeo: an IMEO model already trained,
    classfier: a classifier, already created to train,
    tr, tr_o: training data and lables
    vl, vl_o: validaiton data and lables
    lr: the learning rate used for the joint model composed by imeo and classifier
    ep: the max number of epoch to train the joint model
    wd: weight decay
    batch_size: the size of the batch
    patience: how much time to wait beforse stopping giving no improvements
    print_time: a print will be done every print_time epochs
    '''
    imeo.unfreeze()
    fullModel = Sequential(imeo.encoder,classifier)
    history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    train_loader = DataLoader(TensorDataset(tr, tr_o), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(vl, vl_o), batch_size=batch_size)
    fullModel.to(device)
    fullModel.train()
    optimizer = torch.optim.Adam(fullModel.parameters(), lr=lr, weight_decay=wd)
    print('Training jointly models')
    for epoch in range(ep):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        epoch_train_accuracy = 0.0
        epoch_val_accuracy = 0.0
        #train
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            x = imeo.set_mask_for_imputation(x)

            y_hat:torch.Tensor = fullModel(x).squeeze()
            loss = binary_cross_entropy(y_hat, y)
            epoch_train_accuracy += binary_accuracy(y_hat, y)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        #evaluate
        with torch.no_grad():
            fullModel.eval()
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                x = imeo.set_mask_for_imputation(x)

                y_hat = fullModel(x).squeeze()
                loss = binary_cross_entropy(y_hat, y)
                epoch_val_accuracy += binary_accuracy(y_hat, y)
                epoch_val_loss += loss.item()

        history['train_loss'].append(epoch_train_loss / len(train_loader))
        history['val_loss'].append(epoch_val_loss / len(val_loader))
        history['train_acc'].append(epoch_train_accuracy / len(train_loader))
        history['val_acc'].append(epoch_val_accuracy / len(val_loader))

        if (epoch + 1) % print_time == 0:
                print(f"Epoch {epoch+1}/{ep}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, Val Accuracy: {epoch_val_accuracy/len(val_loader):.4f}")
        
        #check stopping criteria
        if patience > 0:
            if epoch > patience:
                if history['val_loss'][-patience] < history['val_loss'][-1]*0.999:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
    return history