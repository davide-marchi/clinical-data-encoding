import torch
import torch.nn as nn
from torcheval.metrics.functional import r2_score

torch.manual_seed(42)

# TODO: add num_layers and manage dinamically construction of the encoder and decoder

class IMEO(nn.Module):
    def __init__(self, inputSize:int, total_binary_columns: int = 0, embedding_percentage: float = 0.25):
        '''
        inputSize: int the number of data columns + mask columns (should be an even number)
        total_binary_columns: int the number of binary columns in the data (binary data should be the first columns)
        embedding_dim: int the number of neurons in the embedding layer
        '''
        super().__init__()
        self.total_binary_columns = total_binary_columns
        self.inputSize = inputSize
        self.imputation = None
        self.embedding_dim = int(inputSize * embedding_percentage / 2) # 25% of HALF of the input size (mask is not counted)
        neurons_num = []
        # TODO: here range 3 is number of layers - 1 and 4 is the number of layers
        for i in range(3):
            #                                      difference of neurons between each layer
            neurons_num.append(int(inputSize - (i+1)*((inputSize - self.embedding_dim) / 4)))
        # TODO: make a for with encoder.append or encoder.add_module 
        self.encoder = nn.Sequential(
            nn.Linear(in_features=inputSize, out_features=neurons_num[0], bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(neurons_num[0]),
            nn.Linear(in_features=neurons_num[0], out_features=neurons_num[1], bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(neurons_num[1]),
            nn.Linear(in_features=neurons_num[1], out_features=neurons_num[2], bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(neurons_num[2]),
            nn.Linear(in_features=neurons_num[2], out_features=self.embedding_dim, bias=True),
            nn.LeakyReLU(),
        )
        # TODO: make a for with encoder.append or encoder.add_module
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(self.embedding_dim),
            nn.Linear(in_features=self.embedding_dim, out_features=neurons_num[2], bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(neurons_num[2]), 
            nn.Linear(in_features=neurons_num[2], out_features=neurons_num[1], bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(neurons_num[1]),
            nn.Linear(in_features=neurons_num[1], out_features=neurons_num[0], bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(neurons_num[0]),
            nn.Linear(in_features=neurons_num[0], out_features=neurons_num[0], bias=True),
            nn.LeakyReLU()
        )
        self.binaOut = nn.Linear(in_features=neurons_num[0], out_features=total_binary_columns, bias=True)
        self.binActivation = nn.Sigmoid()
        self.contOut = nn.Linear(in_features=neurons_num[0], out_features=inputSize//2 - total_binary_columns, bias=True)
        #initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.uniform_(m.bias, -0.5, 0.5)
        
        print("input size: ", inputSize, 
              "embedding_percentage: ",embedding_percentage,
              "embedding_dim: ", self.embedding_dim, 
              "neurons_num: ", neurons_num)
    
    def saveModel(self, path):
        torch.save(self, path)

    def loadModel(self, path):
        self.load_state_dict(torch.load(path))

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.uniform_(m.bias, -0.5, 0.5)

    def forward(self, x):
        '''
        x: tensor of shape (batch_size, inputSize)
        return: tensor of shape (batch_size, inputSize//2)
        '''
        x = self.encoder(x)
        x = self.decoder(x)
        x1 = self.binActivation(self.binaOut(x))
        x2 = self.contOut(x)
        x = torch.cat((x1, x2), dim=1)
        return x
    
    def encode(self, x):
        '''
        x: tensor of shape (batch_size, inputSize)
        return: tensor of shape (batch_size, embedding_dim)
        '''
        return self.encoder(x)
    
    def compute_r_squared(self, batch, binCol: int = 0):
        '''
        batch: tensor of shape (batch_size, inputSize)
        binCol: int the number of binary columns in the data (binary data should be the first columns)
            if omitted, the total_binary_columns used during creation of the NN will be used
        return: float the r squared value
        '''
        x = batch
        val, mask = torch.chunk(x, 2, dim=1)
        y_hat = torch.mul(self(x), mask)
        y = torch.mul(val, mask)
        if self.total_binary_columns != 0 or binCol != 0:
            binCol = self.total_binary_columns if binCol == 0 else binCol
            y_hat = y_hat[:, :-binCol]
            y = y[:, :-binCol]
        return r2_score(y, y_hat).item()
    
    def compute_accuracy(self, batch, binCol: int = 0):
        '''
        batch: tensor of shape (batch_size, inputSize)
        binCol: int the number of binary columns in the data (binary data should be the first columns)
            if omitted, the total_binary_columns used during creation of the NN will be used
        return: float the accuracy value
        '''
        if self.total_binary_columns == 0 and binCol == 0:
            return 0
        x = batch
        binCol = self.total_binary_columns if binCol == 0 else binCol
        val, mask = torch.chunk(x, 2, dim=1)
        y_hat = torch.mul(self(x), mask)
        y = torch.mul(val, mask)
        y_hat_bin = y_hat[:, :binCol]
        y_bin = y[:, :binCol]
        y_hat_bin = torch.round(y_hat_bin)
        return (torch.sum(y_hat_bin == y_bin) / (y_hat_bin.shape[0] * binCol)).item()
    
    def weighted_measure(self, batch, binCol: int = 0):
        acc = self.compute_accuracy(batch, binCol)
        r2 = self.compute_r_squared(batch, binCol)
        if self.total_binary_columns == 0 and binCol == 0:
            return 0
        binCol = self.total_binary_columns if binCol == 0 else binCol
        binWeight = binCol / (self.inputSize // 2)
        return acc * binWeight + r2 * (1 - binWeight)
    
    def training_step(self, batch, imputation_batch, binaryLossWeight: float = 0.5, forceMSE: bool = False):
        '''
        batch: tensor of shape (batch_size, inputSize)
        binaryLossWeight: float the weight of the binary loss in the total loss (0 <= binaryLossWeight <= 1)
            the weight of the continuous loss will be 1 - binaryLossWeight
        return: the total loss
        '''
        x = imputation_batch
        original_val, missing_mask = torch.chunk(batch, 2, dim=1)
        y_hat = torch.mul(self(x), missing_mask)
        y = torch.mul(original_val, missing_mask)
        if self.total_binary_columns == 0 or forceMSE:
            return nn.functional.mse_loss(y, y_hat)
        y_hat_bin = y_hat[:, :self.total_binary_columns]
        y_bin = y[:, :self.total_binary_columns]
        loss_bin = nn.functional.binary_cross_entropy(y_hat_bin, y_bin)
        y_hat_cont = y_hat[:, :-self.total_binary_columns]
        y_cont = y[:, :-self.total_binary_columns]
        loss_cont = nn.functional.mse_loss(y_cont, y_hat_cont)
        loss = binaryLossWeight*loss_bin + (1 - binaryLossWeight)*loss_cont
        return loss
    
    def compute_metrics(self, batch, binCol: int = 0, metrics:list = ['loss', 'r2', 'acc', 'a2']):
        '''
        batch: tensor of shape (batch_size, inputSize)
        binCol: int the number of binary columns in the data (binary data should be the first columns)
            if omitted, the total_binary_columns used during creation of the NN will be used
        metrics: list the metrics to compute
        return: dict the computed metrics
        '''
        metrics_dict = {}
        if 'loss' in metrics:
            metrics_dict['loss'] = self.training_step(batch).item()
        if 'r2' in metrics:
            metrics_dict['r2'] = self.compute_r_squared(batch, binCol)
        if 'acc' in metrics:
            metrics_dict['acc'] = self.compute_accuracy(batch, binCol)
        if 'a2' in metrics:
            metrics_dict['a2'] = self.weighted_measure(batch, binCol)
        return metrics_dict
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
              
    # sets imputation values to 0
    def mask_imputation_step(self, batch:torch.Tensor, masked_percentage:float = 0.5):
        batch = batch.clone()
        val, mask = torch.chunk(batch, 2, dim=1)
        imputation_mask = torch.where((torch.rand_like(mask) < masked_percentage) & (mask == 1.0), torch.tensor(0.0), mask) 
        # important set to 0 imputation elements in the val part too
        val = torch.mul(val, imputation_mask)
        return torch.cat((val, imputation_mask), dim=1)

    def fit(self,
            train_data:torch.Tensor, 
            val_data:torch.Tensor, 
            optimizer:torch.optim.Optimizer, 
            device:torch.device,
            num_epochs:int = 100, 
            batch_size:int = 64, 
            binary_loss_weight:float = None,
            print_every:int = 200,
            plot:bool = True,
            metrics:list = ['tr_loss', 'val_loss', 'val_r2', 'val_acc', 'val_a2'],
            masked_percentage:float = 0.0,
            early_stopping:int = 0
        ) -> dict:
        
        
        binary_loss_weight = self.total_binary_columns / (self.inputSize // 2) if binary_loss_weight is None else binary_loss_weight
        data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        metrics_hystory = {metric:[] for metric in metrics}
        for i in range(num_epochs):
            for batch in data_loader:
                # fancy imputation stuff
                imputation_batch = self.mask_imputation_step(batch, masked_percentage)
                optimizer.zero_grad()
                batch = batch.to(device)
                imputation_batch = imputation_batch.to(device)
                loss = self.training_step(batch, imputation_batch, binaryLossWeight=binary_loss_weight)
                loss.backward()
                optimizer.step()
            for metric in metrics:
                if metric == 'tr_loss':
                    metrics_hystory[metric].append(self.training_step(train_data,train_data).item())
                elif metric == 'val_loss':
                    metrics_hystory[metric].append(self.training_step(val_data,val_data).item())
                elif metric == 'val_r2':
                    metrics_hystory[metric].append(self.compute_r_squared(val_data, self.total_binary_columns))
                elif metric == 'val_acc':
                    metrics_hystory[metric].append(self.compute_accuracy(val_data, self.total_binary_columns))
                elif metric == 'val_a2':
                    metrics_hystory[metric].append(self.weighted_measure(val_data, self.total_binary_columns))
            if (print_every != 0):
                with torch.no_grad():
                    print(f"Epoch [{i+1}/{num_epochs}]",
                    f"Loss: {metrics_hystory['tr_loss'][-1]:.4f}" if 'tr_loss' in metrics else '',
                    f"Val Loss: {metrics_hystory['val_loss'][-1]:.4f}" if 'val_loss' in metrics else '',
                    f"Val R^2: {metrics_hystory['val_r2'][-1]:.2f}" if 'val_r2' in metrics else '',
                    f"Val acc: {metrics_hystory['val_acc'][-1]:.2f}" if 'val_acc' in metrics else '',
                    f"New a^2: {metrics_hystory['val_a2'][-1]:.2f}" if 'val_a2' in metrics else '',
                    "      ", end='\r')
            if print_every != 0 and i % print_every == 0:
                print('')
                
            if early_stopping > 0:
                if i > early_stopping:
                    if metrics_hystory['val_loss'][-early_stopping] < metrics_hystory['val_loss'][-1]*0.999:
                        print(f"\nEarly stopping at epoch {i+1}")
                        break
                    
        print('\nTraining complete\t\t\t\t')
        if plot:
            import matplotlib.pyplot as plt
            for metric in metrics:
                if 'loss' in metric:
                    plt.plot(metrics_hystory[metric], label=metric)
            plt.show()
            for metric in metrics:
                if 'loss' not in metric:
                    plt.plot(metrics_hystory[metric], label=metric)
            plt.show()
        return metrics_hystory
        