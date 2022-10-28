from helper import *
from model import *

class Runner(object):
    def __init__(self, params):
        self.p = params
        if self.p.device != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda:'+self.p.device)
        else:
            self.device = torch.device('cpu')
        # prepare data
        print("device already: ", self.device)
        self.dataset = self.p.dataset
        self.data = load_dataset(self.dataset)
        if isinstance(self.data, tuple):
            self.variable = self.data[1]
            self.data = self.data[0]
        else:
            self.variable = self.p.tv
        self.p.vn = self.data.shape[1]
        print("the shape of dataset <" + self.dataset + "> is", self.data.shape)
        self.PreX, self.PreY = get_sample(self.data, self.p.ttl, self.variable, self.p.tpt, self.p.ebl)
        self.train_iter, self.test_iter = self.get_iter()
        # prepare model
        if self.p.restore == False:
            self.p.save_name = self.p.dataset + '[' + str(self.p.tv) + '](' + self.p.model + ')' + time.strftime('%Y%m%d') + '_' + time.strftime('%H:%M:%S')
        self.save_name = self.p.save_name
        self.model = self.add_model()
        self.optimizer = self.add_optimizer()
        self.loss_function = nn.MSELoss().to(self.device)
        if self.p.restore:
            self.load_model(self.save_name)
            self.Indicators = self.load_ind(self.save_name)
        else:
            self.Indicators = init_indicators()
        show_params(self.p)

    def get_iter(self):
        # get iter for training
        train_indexs, test_indexs = get_index(self.p.si, self.p.ei, self.p.itv, self.p.tr, self.p.seed, self.p.dr)
        train_x = []
        train_y = []
        for train_index in train_indexs:
            x, y = get_sample(self.data, train_index, self.variable, self.p.tpt, self.p.ebl)
            train_x.append(x)
            train_y.append(y)
        test_x = []
        test_y = []
        for test_index in test_indexs:
            x, y = get_sample(self.data, test_index, self.variable, self.p.tpt, self.p.ebl)
            test_x.append(x)
            test_y.append(y)
        np.save('../dataset/'+self.dataset+'/'+self.dataset+str(self.p.tv)+'_train_x', train_x) 
        np.save('../dataset/'+self.dataset+'/'+self.dataset+str(self.p.tv)+'_train_y', train_y)
        np.save('../dataset/'+self.dataset+'/'+self.dataset+str(self.p.tv)+'_test_x', test_x) 
        np.save('../dataset/'+self.dataset+'/'+self.dataset+str(self.p.tv)+'_test_y', test_y)
        train_x = np.load('../dataset/'+self.dataset+'/'+self.dataset+str(self.p.tv)+'_train_x.npy') 
        train_y = np.load('../dataset/'+self.dataset+'/'+self.dataset+str(self.p.tv)+'_train_y.npy') 
        test_x = np.load('../dataset/'+self.dataset+'/'+self.dataset+str(self.p.tv)+'_test_x.npy') 
        test_y = np.load('../dataset/'+self.dataset+'/'+self.dataset+str(self.p.tv)+'_test_y.npy')
        train_x = torch.tensor(train_x)
        train_y = torch.tensor(train_y)
        test_x = torch.tensor(test_x)
        test_y = torch.tensor(test_y)
        train_set = torch.utils.data.TensorDataset(train_x, train_y)
        test_set = torch.utils.data.TensorDataset(test_x, test_y)
        train_iter = torch.utils.data.DataLoader(train_set, batch_size=self.p.bs,
                                                shuffle=True)
        test_iter = torch.utils.data.DataLoader(test_set, batch_size=self.p.bs,
                                                shuffle=False)
        return train_iter, test_iter
    
    def add_model(self):
        # add a new model
        model_name = self.p.model
        if model_name == "LSTM":
            return MLSTM(self.p).double().to(self.device)
        elif model_name == "Smodule":
            self.p.sunits.append(self.p.ebl)
            return Smodule(self.p).double().to(self.device)
        elif model_name == "Tmodule":
            self.p.tunits.append(self.p.ebl)
            return Tmodule(self.p).double().to(self.device)
        elif model_name == "STSM":
            return STSM(self.p).double().to(self.device)

    def add_optimizer(self):
        # add a new optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.w)
        return optimizer

    def save_model(self, model_name):
        # save current model
        path = os.path.join('./models', model_name)
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)

    def load_model(self, model_name):
        # load a saved model
        path = os.path.join('./models', model_name)
        state = torch.load(path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

    def save_ind(self, model_name):
        # save current indicators
        np.save(os.path.join('./indicators', model_name), self.Indicators)

    def load_ind(self, model_name):
        # load saved indicators
        return np.load('./indicators/'+model_name+'.npy', allow_pickle=True).item()

    def get_loss(self, train_loader, if_train):
        # compute loss
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            outputs = self.model(x)
            loss = self.loss_function(outputs, y)
            if if_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()
            if if_train:
                if (i + 1) % self.p.infitv == 0:
                    print (("Step [{}/{}] Train Loss: {:.4f} "+self.p.save_name)
                    .format(i+1, len(train_loader), loss.item()))
            else:
                print (("Step [{}/{}] Test Loss: {:.4f} "+self.p.save_name)
                    .format(i+1, len(train_loader), loss.item()))
        return total_loss / len(train_loader)

    def fit(self):
        # train the model
        print(self.model)
        for epoch in range(len(self.Indicators['train_losses']), self.p.epoch):
            print('='*5 + ' epoch ' + str(epoch+1) + '='*5)
            self.model.train()
            loss = self.get_loss(self.train_iter, if_train=True)
            self.Indicators['train_losses'].append(loss)
            self.model.eval()
            Pred = self.model(torch.tensor(self.PreX).unsqueeze(0).to(self.device)).squeeze()
            if len(self.test_iter) > 0:
                loss = self.get_loss(self.test_iter, if_train=False)
                self.Indicators['valid_losses'].append(loss)
            if self.p.cl and epoch%40 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.9
        self.model.eval()
        Pred = self.model(torch.tensor(self.PreX).unsqueeze(0).to(self.device)).squeeze()
        compare_results(embedding_resolve(torch.tensor(self.PreY)), embedding_resolve(Pred), self.p.tpt, self.p.model, self.p.dataset)
        show_curve(self.Indicators['train_losses'], 'EPOCH', 'loss value', 'loss curve for training')
        show_curve(self.Indicators['valid_losses'], 'EPOCH', 'loss value', 'loss curve for validating')
        self.save_model(self.save_name)
        self.save_ind(self.save_name)

    def predict(self):
        # predict the value
        self.model.eval()
        Pred = self.model(torch.tensor(self.PreX).unsqueeze(0).to(self.device)).squeeze()
        lresult = embedding_resolve(torch.tensor(self.PreY))
        yresult = embedding_resolve(Pred)
        for new_index in range(self.p.ebl-1, self.p.prelen, self.p.ebl-1):
            x, y = get_sample(self.data, self.p.ttl+new_index, self.variable, self.p.tpt, self.p.ebl)
            x[:, self.variable] = yresult[new_index:new_index+self.p.tpt]
            lresult = np.hstack((lresult, embedding_resolve(torch.tensor(y))[self.p.tpt:]))
            yresult = np.hstack((yresult, embedding_resolve(self.model(torch.tensor(x).unsqueeze(0).to(self.device)).squeeze())[self.p.tpt:]))
        lresult = lresult[:self.p.tpt+self.p.prelen]
        yresult = yresult[:self.p.tpt+self.p.prelen]
        compare_results(lresult, yresult, self.p.tpt, self.p.model, self.p.dataset)

    def run(self):
        # decide to train or preidct
        if self.p.prelen:
            self.predict()
        else:
            self.fit()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = init_parser()
    args = parser.parse_args()
    remove_random(args.seed)
    model = Runner(args)
    model.run()