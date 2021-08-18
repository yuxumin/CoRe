import torch

class Group_helper(object):
    def __init__(self, dataset, depth, Symmetrical = True, Max = None, Min = None):
        '''
            dataset : list of deltas (CoRe method) or list of scores (RT method)
            depth : depth of the tree
            Symmetrical: (bool) Whether the group is symmetrical about 0.
                        if symmetrical, dataset only contains th delta bigger than zero.
            Max : maximum score or delta for a certain sports.
        '''
        self.dataset = sorted(dataset)
        self.length = len(dataset)
        self.num_leaf = 2**(depth-1)
        self.symmetrical = Symmetrical
        self.max = Max
        self.min = Min
        self.Group = [[] for _ in range(self.num_leaf)]
        self.build()

    def build(self):
        '''
            separate region of each leaf
        '''
        if self.symmetrical:
            # delta in dataset is the part bigger than zero.
            for i in range(self.num_leaf // 2):
                # bulid positive half first
                Region_left = self.dataset[int( (i / (self.num_leaf//2)) * (self.length-1) )]
                
                if i == 0:
                    if self.min != None:
                        Region_left = self.min
                    else:
                        Region_left = self.dataset[0]
                Region_right = self.dataset[int( ( (i + 1) /(self.num_leaf//2)) * (self.length-1) )]
                if i == self.num_leaf//2 - 1:
                    if self.max != None:
                        Region_right = self.max 
                    else:
                        Region_right = self.dataset[-1]
                self.Group[self.num_leaf // 2 + i] = [Region_left,Region_right]
            for i in range(self.num_leaf // 2):
                self.Group[i] = [-i for i in self.Group[self.num_leaf - 1 -i ]]
            for group in self.Group:
                group.sort()
        else:
            for i in range(self.num_leaf):
                Region_left = self.dataset[int( (i / self.num_leaf) * (self.length-1) )]
                if i == 0:
                    if self.min != None:
                        Region_left = self.min
                    else:
                        Region_left = self.dataset[0]
                Region_right = self.dataset[int( ( (i + 1) / self.num_leaf) * (self.length-1) )]
                if i == self.num_leaf - 1:
                    if self.max != None:
                        Region_right = self.max 
                    else:
                        Region_right = self.dataset[-1]
                self.Group[i] = [Region_left,Region_right]
    def produce_label(self, scores):
        if isinstance(scores,torch.Tensor):
            scores = scores.detach().cpu().numpy().reshape(-1,)
        glabel = []
        rlabel = []
        for i in range(self.num_leaf):
            # if in one leaf : left == right
            # we should treat this leaf differently
            leaf_cls = []
            laef_reg = []
            for score in scores:
                if score >= 0 and (score < self.Group[i][1] and score >= self.Group[i][0]):
                    leaf_cls.append(1)
                elif score < 0 and (score <= self.Group[i][1] and score > self.Group[i][0]):
                    leaf_cls.append(1)
                else:
                    leaf_cls.append(0)

                if leaf_cls[-1] == 1:
                    if self.Group[i][1] == self.Group[i][0]:
                        rposition = score - self.Group[i][0]
                    else:
                        rposition =  (score - self.Group[i][0])/(self.Group[i][1] - self.Group[i][0])
                else:
                    rposition = -1
                laef_reg.append(rposition)
            glabel.append(leaf_cls)
            rlabel.append(laef_reg)
        glabel =  torch.tensor(glabel).cuda()
        rlabel =  torch.tensor(rlabel).cuda()
        return glabel , rlabel
    def inference(self, probs, deltas):
        '''
            probs: bs * leaf
            delta: bs * leaf
        '''
        predictions = []
        for n in range(probs.shape[0]):
            prob = probs[n] 
            delta = deltas[n]
            leaf_id = prob.argmax()
            if self.Group[leaf_id][0] == self.Group[leaf_id][1]:
                prediction = self.Group[leaf_id][0] + delta[leaf_id]
            else:
                prediction =  self.Group[leaf_id][0] + (self.Group[leaf_id][1] -  self.Group[leaf_id][0]) * delta[leaf_id]
            predictions.append(prediction)
        return torch.tensor(predictions).reshape(-1,1)
    def get_Group(self):
        return self.Group

    def number_leaf(self):
        return self.num_leaf