import sys
sys.path.append('..')
import utils
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import BertModel

class BertNERModel(utils.framework.FewShotNERModel):
    """_summary_
    A baseline model using only bert with one additional output 
    layer on top of BERT output.

    This class inherits the FewShotNERModel
    Args:
        utils (_type_): _description_
    """

    def __init__(self, args, word_encoder, dot=False, ignore_index=-1):
        utils.framework.FewShotNERModel.__init__(self, args, word_encoder, ignore_index=ignore_index)
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.1) 
        # output layer
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 111)# 111 number of classes
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, ids, mask):

        outputs = self.bert(ids, attention_mask=mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        logits = torch.cat(logits, 0)

        pred = torch.argmax(self.softmax(logits), dim=1)
        return logits, pred


