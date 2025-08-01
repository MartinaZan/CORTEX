import numpy as np
import torch

from src.core.oracle_base import Oracle
from src.core.torch_base import TorchBase
from src.dataset.utils.dataset_torch import TorchGeometricDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class OracleTorch(TorchBase, Oracle):
                                
            
    def real_fit(self):
        super().real_fit()
        self.evaluate(self.dataset, fold_id=self.fold_id)
            
    @torch.no_grad()
    def evaluate(self, dataset, fold_id=0):
        loader = dataset.get_torch_loader(fold_id=fold_id, batch_size=self.batch_size, usage='test')
        
        losses = []
        labels_list, preds = [], []
        for batch in loader:
            batch.batch = batch.batch.to(self.device)
            node_features = batch.x.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            edge_weights = batch.edge_attr.to(self.device)
            labels = batch.y.to(self.device).long()
            
            self.optimizer.zero_grad()  
            pred = self.model(node_features, edge_index, edge_weights, batch.batch)            
            loss = self.loss_fn(pred, labels)
            losses.append(loss.to('cpu').detach().numpy())
            
            labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
            preds += list(pred.squeeze().detach().to('cpu').numpy())
            
        accuracy = self.accuracy(labels_list, preds)
        self.context.logger.info(f'Test Accuracy = {np.mean(accuracy):.4f}')

        # Other metrics
        pred_classes = [np.argmax(p) if isinstance(p, (list, np.ndarray)) else p for p in preds]
        f1 = f1_score(labels_list, pred_classes, average='macro')
        precision = precision_score(labels_list, pred_classes, average='macro')
        recall = recall_score(labels_list, pred_classes, average='macro')
        self.context.logger.info(f'Test F1 Score = {f1:.4f}')
        self.context.logger.info(f'Test Precision = {precision:.4f}')
        self.context.logger.info(f'Test Recall = {recall:.4f}')
        

    def _real_predict(self, data_instance, return_embeddings=False):

        if return_embeddings:
            output, embeddings = self._real_predict_proba(data_instance,return_embeddings)
            output = torch.argmax(output).item()
            return output, embeddings

        return torch.argmax(self._real_predict_proba(data_instance,return_embeddings)).item()

    @torch.no_grad()
    def _real_predict_proba(self, data_inst, return_embeddings=False):
        data_inst = TorchGeometricDataset.to_geometric(data_inst)
        node_features = data_inst.x.to(self.device)
        edge_index = data_inst.edge_index.to(self.device)
        edge_weights = data_inst.edge_attr.to(self.device)

        if return_embeddings:
            output, embeddings = self.model(node_features,edge_index,edge_weights, None, return_embeddings=True)
            output = output.cpu().squeeze()
            return output, embeddings
        
        return self.model(node_features,edge_index,edge_weights, None).cpu().squeeze()
                     
    def check_configuration(self):#TODO: revise configuration
        super().check_configuration()
        local_config = self.local_config

        if 'model' not in local_config['parameters']:
            local_config['parameters']['model'] = {
                'class': "src.oracle.nn.gcn.DownstreamGCN",
                "parameters" : {}
            }

        # set defaults
        local_config['parameters']['model']['parameters']['node_features'] = self.dataset.num_node_features()
        local_config['parameters']['model']['parameters']['n_classes'] = self.dataset.num_classes