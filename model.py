from helper import *

class MLSTM(nn.Module):
    # LSTM model
    def __init__(self, paras):
        super(MLSTM, self).__init__()
        self.units = paras.fcunits
        self.mylstm = nn.LSTM(self.units[0], self.units[0]*4, 2, bidirectional=paras.bi, batch_first = True)
        self.relu1 = nn.ReLU()
        if paras.act == 0:
            self.activate = nn.ReLU()
        else:
            self.activate = nn.Tanh()
        self.layers = self._make_layer(paras.dp, paras.tpt, paras.ebl)

    def forward(self, input):
        predict, _ = self.mylstm(input)
        predict = predict.squeeze(1)
        predict = self.relu1(predict)
        return self.layers(predict)

    def _make_layer(self, mdropout, train_time_points, embedding_length):
        layers = []
        layers += [nn.Linear(self.units[0]*4, self.units[1]), nn.BatchNorm1d(train_time_points), self.activate, nn.Dropout(p=mdropout)]
        for i in range(1, len(self.units)-1):
            layers += [nn.Linear(self.units[i], self.units[i+1]), nn.BatchNorm1d(train_time_points), self.activate, nn.Dropout(p=mdropout)]
        layers += [nn.Linear(self.units[-1], embedding_length)]
        return nn.Sequential(*layers)

class Smodule(nn.Module):
    # Single Spatial module
    def __init__(self, paras):
        super(Smodule, self).__init__()
        self.units = paras.sunits
        if paras.act == 0:
            self.activate = nn.ReLU()
        else:
            self.activate = nn.Tanh()
        self.layers = self._make_layer(paras.dp, paras.tpt)
        
    def _make_layer(self, DROPOUT, train_time_points):
        layers = []
        for i in range(len(self.units)-2):
            layers.append(nn.Sequential(*[nn.Linear(self.units[i], self.units[i+1]), nn.BatchNorm1d(train_time_points), self.activate, nn.Dropout(p=DROPOUT)]))
        layers += [nn.Linear(self.units[-2], self.units[-1])]
        return nn.Sequential(*layers)
    
    def forward(self, input):
        predict = self.layers(input)
        return predict

class Tmodule(nn.Module):
    # Single Temporal module
    def __init__(self, paras):
        super(Tmodule, self).__init__()
        self.kernel_widths = paras.kw
        self.kernel_nums = paras.kn
        self.kernel_height = paras.vn
        self.units = paras.tunits
        self.dropout = paras.dp
        self.Convs = self.make_convs()
        self.pactivate = nn.ReLU()
        if paras.act == 0:
            self.activate = nn.ReLU()
        else:
            self.activate = nn.Tanh()
        self.Fcs = self.make_fcs()
        self.layers = self._make_layer()
        
    def make_convs(self):
        Convs = []
        for i in range(len(self.kernel_widths)):
            Convs.append(nn.Conv1d(1, self.kernel_nums[i], (self.kernel_widths[i], self.kernel_height)))
        return nn.Sequential(*Convs)
    
    def make_fcs(self):
        Fcs = []
        for i in range(len(self.kernel_nums)):
            Fcs.append(nn.Linear(np.sum(self.kernel_nums[0:i+1]), self.units[0]))
        return nn.Sequential(*Fcs)
    
    def _make_layer(self):
        layers = []
        for i in range(len(self.units)-2):
            layers.append(nn.Sequential(*[nn.Linear(self.units[i], self.units[i+1]), nn.BatchNorm1d(1), self.activate, nn.Dropout(p=self.dropout)]))
        layers += [nn.Linear(self.units[-2], self.units[-1])]
        return nn.Sequential(*layers)
    
    def act_and_pool(self, x):
        out = self.pactivate(x)
        out = nn.functional.max_pool2d(out, (out.size(2), out.size(3)))
        out = out.squeeze(2).squeeze(2)
        return out
    
    def one_step(self,x):
        use_convs = []
        seq_length = x.shape[1]
        for i in range(len(self.kernel_widths)):
            if self.kernel_widths[i] <= seq_length:
                use_convs.append(self.Convs[i])
        x = x.unsqueeze(1)
        output = torch.cat([self.act_and_pool(use_convs[i](x)) for i in range(len(use_convs))], 1)
        f_index = np.min((len(self.kernel_nums)-1, seq_length-1))
        output = self.Fcs[f_index](output).unsqueeze(1)
        return self.layers(output)
    
    def forward(self,x):
        seq_length = x.shape[1]
        output = [self.one_step(x[:,:i+1,:]) for i in range(0, seq_length)]
        return torch.cat(output,1)

class STSM(nn.Module):
    def __init__(self, paras):
        # merge_mode为1相加，为0是连接
        super(STSM, self).__init__()
        self.dataset = paras.dataset
        self.Tmodule = Tmodule(paras)
        self.Smodule = Smodule(paras)
        self.merge_mode = paras.mm
        self.embedding_size = paras.ebl
        self.train_time_points = paras.tpt
        self.dropout = paras.dp
        self.tunits = paras.tunits
        self.sunits = paras.sunits
        self.fcunits = paras.fcunits
        if paras.act == 0:
            self.activate = nn.ReLU()
        else:
            self.activate = nn.Tanh()
        self.layers = self._make_layer()        
        
    def _make_layer(self):
        layers = []
        if self.merge_mode == 0:
            # [temporal, spatial, global]
            layers.append(nn.Sequential(*[nn.Linear(self.tunits[-1]+self.sunits[-1]+self.sunits[0], self.fcunits[0]), nn.BatchNorm1d(self.train_time_points), self.activate, nn.Dropout(p=self.dropout)]))
        elif self.merge_mode == 1:
            # [temporal, spatial], without global
            layers.append(nn.Sequential(*[nn.Linear(self.tunits[-1]+self.sunits[-1], self.fcunits[0]), nn.BatchNorm1d(self.train_time_points), self.activate, nn.Dropout(p=self.dropout)]))
        for i in range(len(self.fcunits)-1):
            layers.append(nn.Sequential(*[nn.Linear(self.fcunits[i], self.fcunits[i+1]), nn.BatchNorm1d(self.train_time_points), self.activate, nn.Dropout(p=self.dropout)]))
        if self.dataset == "Solar":
            layers += [nn.Linear(self.fcunits[-1], self.embedding_size), nn.ReLU()]
        else:
            layers += [nn.Linear(self.fcunits[-1], self.embedding_size)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        time_feature = self.Tmodule(x)
        space_feature = self.Smodule(x)
        if self.merge_mode == 0:
            # [temporal, spatial, global]
            minput = torch.cat([time_feature, space_feature, x], 2)
        elif self.merge_mode == 1:
            # [temporal, spatial], without global
            minput = torch.cat([time_feature, space_feature], 2)
        return self.layers(minput)
    