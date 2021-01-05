from pytools import memoize_method
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import modeling_util
import string

class BertRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.BERT_MODEL = 'bert-base-uncased'
        self.CHANNELS = 12 + 1 # from bert-base-uncased
        self.BERT_SIZE = 768 # from bert-base-uncased
        self.bert = BertModel.from_pretrained(self.BERT_MODEL, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            state[key] = state[key].data
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        print("load model : ", path)

    def freeze_bert(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def get_params(self):
        params = [(k, v) for k, v in self.named_parameters() if v.requires_grad]
        non_bert_params = [v for k, v in params if not k.startswith('bert')]
        bert_params = [v for k, v in params if k.startswith('bert')]
        return non_bert_params, bert_params 

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 3 # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings

        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0 # remove padding (will be masked anyway)

        # execute BERT model
        #result = self.bert(toks, segment_ids.long(), mask)
        result_tuple = self.bert(toks, mask, segment_ids.long())
        result = result_tuple[2] ## all hidden_states

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:QLEN+1] for r in result]
        doc_results = [r[:, QLEN+2:-1] for r in result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i*BATCH:(i+1)*BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        return cls_results, query_results, doc_results

    def encode_colbert(self, query_tok, query_mask, doc_tok, doc_mask):
        # encode without subbatching
        BATCH, QLEN = query_tok.shape
        DIFF = 5 # = [CLS], 2x[SEP], [Q], [D]
        maxlen = self.bert.config.max_position_embeddings

        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        query_toks = query_tok
        # query_mask = query_mask
        doc_toks = doc_tok[:, :MAX_DOC_TOK_LEN]
        doc_mask = doc_mask[:, :MAX_DOC_TOK_LEN]
        
        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        Q_tok = torch.full(
            size=(BATCH, 1), fill_value=1, dtype=torch.long
        ).cuda()  # [unused0] = 1
        D_tok = torch.full(
            size=(BATCH, 1), fill_value=2, dtype=torch.long
        ).cuda()  # [unused1] = 2

        # Query augmentation with [MASK] tokens ([MASK] = 103)
        query_toks[query_toks == -1] = torch.tensor(103).cuda()
        query_mask = torch.ones_like(query_mask)

        # build BERT input sequences
        toks = torch.cat([CLSS, Q_tok, query_toks, SEPS, D_tok, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, ONES, query_mask, ONES, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (3+QLEN) + [ONES] * (2+doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0 # remove padding (will be masked anyway)
        
        # modifiy doc_mask
        doc_mask = torch.cat([ONES, doc_mask, ONES], dim=1)

        # execute BERT model
        result_tuple = self.bert(toks, mask, segment_ids.long())
        result = result_tuple[2] ## all hidden_states

        # extract relevant subsequences for query and doc
        query_results = [r[:, :QLEN+3] for r in result]
        doc_results = [r[:, QLEN+3:] for r in result]

        return query_results, query_mask, doc_results, doc_mask

class TwoBertRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.BERT_MODEL = 'bert-base-uncased'
        self.CHANNELS = 12 + 1 # from bert-base-uncased
        self.BERT_SIZE = 768 # from bert-base-uncased
        self.bert = BertModel.from_pretrained(self.BERT_MODEL, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            state[key] = state[key].data
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        print("load model : ", path)

    def freeze_bert(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def get_params(self):
        params = [(k, v) for k, v in self.named_parameters() if v.requires_grad]
        non_bert_params = [v for k, v in params if not k.startswith('bert')]
        bert_params = [v for k, v in params if k.startswith('bert')]
        return non_bert_params, bert_params 

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 3 # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences query & doc
        q_toks = torch.cat([CLSS, query_toks, SEPS], dim=1)
        q_mask = torch.cat([ONES, query_mask, ONES], dim=1)
        q_segid = torch.cat([NILS] * (2+QLEN), dim=1)
        q_toks[q_toks == -1] = 0

        d_toks = torch.cat([CLSS, doc_toks, SEPS], dim=1)
        d_mask = torch.cat([ONES, doc_mask, ONES], dim=1)
        d_segid = torch.cat([NILS] * (2+doc_toks.shape[1]), dim=1)
        d_toks[d_toks == -1] = 0

        # execute BERT model
        q_result_tuple = self.bert(q_toks, q_mask, q_segid.long())
        d_result_tuple = self.bert(d_toks, d_mask, d_segid.long())
        q_result = q_result_tuple[2]
        d_result = d_result_tuple[2]

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:-1] for r in q_result]
        doc_results = [r[:, 1:-1] for r in d_result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        q_cls_results = []
        for layer in q_result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i*BATCH:(i+1)*BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            q_cls_results.append(cls_result)

        d_cls_results = []
        for layer in d_result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i*BATCH:(i+1)*BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            d_cls_results.append(cls_result)

        return q_cls_results, d_cls_results, query_results, doc_results

    def encode_colbert(self, query_tok, query_mask, doc_tok, doc_mask):
        # encode without subbatching
        query_lengths = (query_mask > 0).sum(1)
        doc_lengths = (doc_mask > 0).sum(1)
        BATCH, QLEN = query_tok.shape
        # QLEN : 20
        # DIFF = 2  # = [CLS] and [SEP]
        maxlen = self.bert.config.max_position_embeddings
        # MAX_DOC_TOK_LEN = maxlen - DIFF  # doc maxlen: 510

        doc_toks = F.pad(doc_tok[:, : maxlen - 2], pad=(0, 1, 0, 0), value=-1)
        doc_mask = F.pad(doc_mask[:, : maxlen - 2], pad=(0, 1, 0, 0), value=0)
        query_toks = query_tok

        query_lengths = torch.where(query_lengths > 19, torch.tensor(19).cuda(), query_lengths)
        query_toks[torch.arange(BATCH), query_lengths] = self.tokenizer.vocab["[SEP]"]
        query_mask[torch.arange(BATCH), query_lengths] = 1
        doc_lengths = torch.where(doc_lengths > 510, torch.tensor(510).cuda(), doc_lengths)
        doc_toks[torch.arange(BATCH), doc_lengths] = self.tokenizer.vocab["[SEP]"]
        doc_mask[torch.arange(BATCH), doc_lengths] = 1

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab["[CLS]"])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab["[SEP]"])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences query & doc
        q_toks = torch.cat([CLSS, query_toks], dim=1)
        q_mask = torch.cat([ONES, query_mask], dim=1)
        q_segid = torch.cat([NILS] * (1 + QLEN), dim=1)
        # 2) Query augmentation with [MASK] tokens ([MASK] = 103)
        q_toks[q_toks == -1] = torch.tensor(103).cuda()

        d_toks = torch.cat([CLSS, doc_toks], dim=1)
        d_mask = torch.cat([ONES, doc_mask], dim=1)
        d_segid = torch.cat([NILS] * (1 + doc_toks.shape[1]), dim=1)
        d_toks[d_toks == -1] = 0

        # execute BERT model
        q_result_tuple = self.bert(q_toks, q_mask, q_segid.long())
        d_result_tuple = self.bert(d_toks, d_mask, d_segid.long())
        q_result = q_result_tuple[2]
        d_result = d_result_tuple[2]

        # extract relevant subsequences for query and doc
        query_results = [r[:, :] for r in q_result]  # missing representation for cls and sep?
        doc_results = [r[:, :] for r in d_result]

        return query_results, q_mask, doc_results, d_mask

class VanillaBertRanker(BertRanker):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        return self.cls(self.dropout(cls_reps[-1]))

class TwinBertRanker(TwoBertRanker):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        q_cls_reps, d_cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        q_cls_rep = F.normalize(q_cls_reps[-1], p=2, dim=0)
        d_cls_rep = F.normalize(d_cls_reps[-1], p=2, dim=0)
        score = F.cosine_similarity(q_cls_rep, d_cls_rep)
        #print(score)
        return score

class TwinBertResRanker(TwoBertRanker):
    def __init__(self, qd=True):
        super().__init__()
        self.qd = qd
        self.dropout = torch.nn.Dropout(0.1)
        self.wpool = torch.nn.AdaptiveAvgPool2d((1,self.BERT_SIZE))
        self.res = torch.nn.Linear(self.BERT_SIZE, self.BERT_SIZE)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        q_cls_reps, d_cls_reps, q_reps, d_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)

        if(self.qd):
            x1 = self.wpool(q_reps[-1]).squeeze(dim=1)
            x2 = self.wpool(d_reps[-1]).squeeze(dim=1)
        else:
            x1 = q_cls_reps[-1]
            x2 = d_cls_reps[-1]

        x = torch.max(x1, x2)
        score = self.cls(self.res(x)+x)
        #print(score)
        return score

class TwinBertSmallRanker(TwoBertRanker):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.cls_q = torch.nn.Linear(self.BERT_SIZE, 256)
        self.cls_d = torch.nn.Linear(self.BERT_SIZE, 256)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        q_cls_reps, d_cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)

        q_cls_small = self.cls_q(self.dropout(q_cls_reps[-1]))
        d_cls_small = self.cls_d(self.dropout(d_cls_reps[-1]))

        q_cls_rep = F.normalize(q_cls_small, p=2, dim=0)
        d_cls_rep = F.normalize(d_cls_small, p=2, dim=0)
        score = F.cosine_similarity(q_cls_rep, d_cls_rep)
        return score

class ColBertRanker(TwoBertRanker):
    def __init__(self):
        super().__init__()
        self.dim = 128  # default: dim=128
        self.skiplist = self.tokenize(string.punctuation)

        self.clinear = torch.nn.Linear(
            self.BERT_SIZE, self.dim, bias=False
        )  # both for queries, documents

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        # q length default: 32  -> 20
        # d length defualt: 180 -> 510

        # 1) Prepend [Q] token to query, [D] token to document
        q_length = query_tok.shape[1]
        d_length = doc_tok.shape[1]
        num_batch_samples = doc_tok.shape[0]

        Q_tok = torch.full(
            size=(num_batch_samples, 1), fill_value=1, dtype=torch.long
        ).cuda()  # [unused0] = 1
        D_tok = torch.full(
            size=(num_batch_samples, 1), fill_value=2, dtype=torch.long
        ).cuda()  # [unused1] = 2
        one_tok = torch.full(size=(num_batch_samples, 1), fill_value=1).cuda()

        query_tok = torch.cat([Q_tok, query_tok[:, : q_length - 1]], dim=1)
        doc_tok = torch.cat([D_tok, doc_tok[:, : d_length - 1]], dim=1)
        query_mask = torch.cat([one_tok, query_mask[:, : q_length - 1]], dim=1)
        doc_mask = torch.cat([one_tok, doc_mask[:, : d_length - 1]], dim=1)

        # 2) Query augmentation with [MASK] tokens ([MASK] = 103)
        q_reps, query_mask, d_reps, doc_mask = self.encode_colbert(
            query_tok, query_mask, doc_tok, doc_mask
        )  # reps includes rep of [CLS], [SEP]
        col_q_reps = self.clinear(q_reps[-1])
        col_d_reps = self.clinear(d_reps[-1])

        # 3) skip punctuations in doc tokens
        cut_doc_tok = torch.cat([one_tok.long(), doc_tok[:, :510], one_tok.long()], dim=1)
        mask = torch.ones_like(doc_mask, dtype=torch.float).cuda()
        mask = torch.where(
            ((cut_doc_tok >= 999) & (cut_doc_tok <= 1013))
            | ((cut_doc_tok >= 1024) & (cut_doc_tok <= 1036))
            | ((cut_doc_tok >= 1063) & (cut_doc_tok <= 1066))
            | (cut_doc_tok == -1),
            torch.tensor(0.0).cuda(),
            doc_mask,
        )
        col_d_reps = col_d_reps * mask.unsqueeze(2)
        q_rep = F.normalize(col_q_reps, p=2, dim=2)
        d_rep = F.normalize(col_d_reps, p=2, dim=2)
        score = (q_rep @ d_rep.permute(0, 2, 1)).max(2).values.sum(1)
        score = score.unsqueeze(1)
        return score

class ColBertVRanker(BertRanker):
    def __init__(self):
        super().__init__()
        self.dim = 128  # default: dim=128
        self.skiplist = self.tokenize(string.punctuation)

        self.clinear = torch.nn.Linear(
            self.BERT_SIZE, self.dim, bias=False
        )  # both for queries, documents

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        # q length default: 32  -> 20 (+ CLS, Q, SEP)
        # d length default: 180 -> 487 (+ D, SEP)

        num_batch_samples = doc_tok.shape[0]
        one_tok = torch.full(size=(num_batch_samples, 1), fill_value=1).cuda()

        q_reps, query_mask, d_reps, doc_mask = self.encode_colbert(
            query_tok, query_mask, doc_tok, doc_mask
        )  # reps includes rep of [CLS], [SEP]
        col_q_reps = self.clinear(q_reps[-1])
        col_d_reps = self.clinear(d_reps[-1])

        # 3) skip punctuations in doc tokens
        cut_doc_tok = torch.cat([one_tok.long(), doc_tok[:, :487], one_tok.long()], dim=1)
        mask = torch.ones_like(doc_mask, dtype=torch.float).cuda()
        mask = torch.where(
            ((cut_doc_tok >= 999) & (cut_doc_tok <= 1013))
            | ((cut_doc_tok >= 1024) & (cut_doc_tok <= 1036))
            | ((cut_doc_tok >= 1063) & (cut_doc_tok <= 1066))
            | (cut_doc_tok == -1),
            torch.tensor(0.0).cuda(),
            doc_mask,
        )
        col_d_reps = col_d_reps * mask.unsqueeze(2)
        q_rep = F.normalize(col_q_reps, p=2, dim=2)
        d_rep = F.normalize(col_d_reps, p=2, dim=2)
        score = (q_rep @ d_rep.permute(0, 2, 1)).max(2).values.sum(1)
        score = score.unsqueeze(1)
        return score

class KnrmBertRanker(TwoBertRanker):
    def __init__(self):
        super().__init__()
        ## KNRM
        MUS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        SIGMAS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        self.kernels = modeling_util.KNRMRbfKernelBank(MUS, SIGMAS)
        #self.batchnorm = torch.nn.BatchNorm1d(self.kernels.count()*self.CHANNELS)
        self.combine = torch.nn.Linear(self.kernels.count() * self.CHANNELS, 1)
        #self.dropout_two = torch.nn.Dropout(0.1).to('cuda:0')
        self.simmat = modeling_util.SimmatModule()

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        q_cls_reps, d_cls_reps, q_reps, d_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
         
        ## knrm bert
        simmat = self.simmat(q_reps, d_reps, query_tok, doc_tok)
        kernels = self.kernels(simmat)
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        simmat = simmat.reshape(BATCH, 1, VIEWS, QLEN, DLEN) \
                       .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN) \
                       .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        result = kernels.sum(dim=3) # sum over document
        mask = (simmat.sum(dim=3) != 0.) # which query terms are not padding?
        result = torch.where(mask, (result + 1e-6).log(), mask.float())
        result = result.sum(dim=2) # sum over query terms
        #result = self.batchnorm(result.sum(dim=2)) # sum over query terms
        #scores = self.combine(self.dropout_two(result)) # linear combination over kernels
        scores = self.combine(result) # linear combination over kernels
        #print(scores)
        return scores

class KnrmVBertRanker(BertRanker):
    def __init__(self):
        super().__init__()
        ## KNRM
        MUS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        SIGMAS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        self.vkernels = modeling_util.KNRMRbfKernelBank(MUS, SIGMAS)
        #self.batchnorm = torch.nn.BatchNorm1d(self.kernels.count()*self.CHANNELS)
        self.vcombine = torch.nn.Linear(self.vkernels.count() * self.CHANNELS, 1)
        #self.vdropout = torch.nn.Dropout(0.1).to('cuda:0')
        self.simmat = modeling_util.SimmatModule()
        #print(self.kernels.count())
        #print(self.kernels.count()*self.CHANNELS)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, q_reps, d_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
         
        ## knrm bert
        simmat = self.simmat(q_reps, d_reps, query_tok, doc_tok)
        kernels = self.vkernels(simmat)
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        simmat = simmat.reshape(BATCH, 1, VIEWS, QLEN, DLEN) \
                       .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN) \
                       .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        result = kernels.sum(dim=3) # sum over document
        mask = (simmat.sum(dim=3) != 0.) # which query terms are not padding?
        result = torch.where(mask, (result + 1e-6).log(), mask.float())
        result = result.sum(dim=2) # sum over query terms
        #result = self.batchnorm(result.sum(dim=2)) # sum over query terms
        #scores = self.vcombine(self.vdropout(result)) # linear combination over kernels
        scores = self.vcombine(result) # linear combination over kernels
        return scores

class DrmmBertRanker(TwoBertRanker):
    def __init__(self):
        super().__init__()
        NBINS = 11
        HIDDEN = 5
        self.simmat = modeling_util.SimmatModule()
        self.histogram = modeling_util.DRMMLogCountHistogram(NBINS)
        self.hidden_1 = torch.nn.Linear(NBINS * self.CHANNELS, HIDDEN)
        self.hidden_2 = torch.nn.Linear(HIDDEN, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        #cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        q_cls_reps, d_cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        histogram = self.histogram(simmat, doc_tok, query_tok)
        BATCH, CHANNELS, QLEN, BINS = histogram.shape
        histogram = histogram.permute(0, 2, 3, 1)
        output = histogram.reshape(BATCH * QLEN, BINS * CHANNELS)
        # repeat cls representation for each query token
        #cls_rep = cls_reps[-1].reshape(BATCH, 1, -1).expand(BATCH, QLEN, -1).reshape(BATCH * QLEN, -1)
        #output = torch.cat([output, cls_rep], dim=1)
        term_scores = self.hidden_2(torch.relu(self.hidden_1(output))).reshape(BATCH, QLEN)
        return term_scores.sum(dim=1).unsqueeze(1)

class DrmmVBertRanker(BertRanker):
    def __init__(self):
        super().__init__()
        NBINS = 11
        HIDDEN = 5
        self.simmat = modeling_util.SimmatModule()
        self.histogram = modeling_util.DRMMLogCountHistogram(NBINS)
        self.hidden_1 = torch.nn.Linear(NBINS * self.CHANNELS, HIDDEN)
        self.hidden_2 = torch.nn.Linear(HIDDEN, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        histogram = self.histogram(simmat, doc_tok, query_tok)
        BATCH, CHANNELS, QLEN, BINS = histogram.shape
        histogram = histogram.permute(0, 2, 3, 1)
        output = histogram.reshape(BATCH * QLEN, BINS * CHANNELS)
        # repeat cls representation for each query token
        #cls_rep = cls_reps[-1].reshape(BATCH, 1, -1).expand(BATCH, QLEN, -1).reshape(BATCH * QLEN, -1)
        #output = torch.cat([output, cls_rep], dim=1)
        term_scores = self.hidden_2(torch.relu(self.hidden_1(output))).reshape(BATCH, QLEN)
        return term_scores.sum(dim=1).unsqueeze(1)

class MultiBertRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.BERT_MODEL = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            state[key] = state[key].data
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        print("load model : ", path)

    def get_params(self):
        params = [(k, v) for k, v in self.named_parameters() if v.requires_grad]
        non_bert_params = [v for k, v in params if '.bert.' not in k ]
        bert_params = [v for k, v in params if '.bert.' in k ]
        return non_bert_params, bert_params 

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

class DuetBertRanker(MultiBertRanker):
    def __init__(self, sub_1, sub_2):
        super().__init__()
        self.bert_1 = sub_1.to('cuda:0')
        self.bert_2 = sub_2.to('cuda:0')
        #2-gpu case
        #self.bert_1 = sub_1.to('cuda:0')
        #self.bert_2 = sub_2.to('cuda:1')

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        score_1 = self.bert_1(query_tok, query_mask, doc_tok, doc_mask)
        #2-gpu case
        #score_2 = self.bert_2(query_tok.to('cuda:1'), query_mask.to('cuda:1'), doc_tok.to('cuda:1'), doc_mask.to('cuda:1'))
        #score_2 = score_2.to('cuda:0')
        score_2 = self.bert_2(query_tok, query_mask, doc_tok, doc_mask)
        score = score_1 + score_2

        return score, score_1, score_2

    def freeze_bert(self):
        self.bert_1.freeze_bert()
        self.bert_2.freeze_bert()

    def load_duet(self, path1, path2):
        print("load duet model")
        self.bert_1.load(path1)
        self.bert_2.load(path2)

class TrioBertRanker(MultiBertRanker):
    def __init__(self, sub_1, sub_2, sub_3):
        super().__init__()
        self.bert_1 = sub_1.to('cuda:0')
        self.bert_2 = sub_2.to('cuda:0')
        self.bert_3 = sub_3.to('cuda:0')
        #2-gpu case
        #self.bert_3 = sub_4.to('cuda:1')

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        score_1 = self.bert_1(query_tok, query_mask, doc_tok, doc_mask)
        score_2 = self.bert_2(query_tok, query_mask, doc_tok, doc_mask)
        score_3 = self.bert_3(query_tok, query_mask, doc_tok, doc_mask)
        #2-gpu case
        #score_3 = self.bert_4(query_tok.to('cuda:1'), query_mask.to('cuda:1'), doc_tok.to('cuda:1'), doc_mask.to('cuda:1')).to('cuda:0')
        score = score_1 + score_2 + score_3 

        return score

    def freeze_bert(self):
        self.bert_1.freeze_bert()
        self.bert_2.freeze_bert()
        self.bert_3.freeze_bert()

    def load_trio(self, path1, path2, path3):
        print("load trio model")
        self.bert_1.load(path1)
        self.bert_2.load(path2)
        self.bert_3.load(path3)

class QuadBertRanker(MultiBertRanker):
    def __init__(self, sub_1, sub_2, sub_3, sub_4):
        super().__init__()
        self.bert_1 = sub_1.to('cuda:0')
        self.bert_2 = sub_2.to('cuda:0')
        self.bert_3 = sub_3.to('cuda:1')
        self.bert_4 = sub_4.to('cuda:1')

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        score_1 = self.bert_1(query_tok, query_mask, doc_tok, doc_mask)
        score_2 = self.bert_2(query_tok, query_mask, doc_tok, doc_mask)
        #2-gpu case
        score_3 = self.bert_3(query_tok.to('cuda:1'), query_mask.to('cuda:1'), doc_tok.to('cuda:1'), doc_mask.to('cuda:1')).to('cuda:0')
        score_4 = self.bert_4(query_tok.to('cuda:1'), query_mask.to('cuda:1'), doc_tok.to('cuda:1'), doc_mask.to('cuda:1')).to('cuda:0')
        score = score_1 + score_2 + score_3 + score_4

        #return score, score_1, score_2, score_3, score_4
        return score

    def freeze_bert(self):
        self.bert_1.freeze_bert()
        self.bert_2.freeze_bert()
        self.bert_3.freeze_bert()
        self.bert_4.freeze_bert()

    def load_quad(self, path1, path2, path3, path4):
        print("load quad model")
        self.bert_1.load(path1)
        self.bert_2.load(path2)
        self.bert_3.load(path3)
        self.bert_4.load(path4)

class HexaBertRanker(MultiBertRanker):
    def __init__(self, sub_1, sub_2, sub_3, sub_4, sub_5, sub_6):
        super().__init__()
        self.bert_1 = sub_1.to('cuda:0')
        self.bert_2 = sub_2.to('cuda:0')
        self.bert_3 = sub_3.to('cuda:0')
        self.bert_5 = sub_4.to('cuda:1')
        self.bert_6 = sub_5.to('cuda:1')
        self.bert_7 = sub_6.to('cuda:1')

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        score_1 = self.bert_1(query_tok, query_mask, doc_tok, doc_mask)
        score_2 = self.bert_2(query_tok, query_mask, doc_tok, doc_mask)
        score_3 = self.bert_3(query_tok, query_mask, doc_tok, doc_mask)
        score_4 = self.bert_4(query_tok.to('cuda:1'), query_mask.to('cuda:1'), doc_tok.to('cuda:1'), doc_mask.to('cuda:1')).to('cuda:0')
        score_5 = self.bert_5(query_tok.to('cuda:1'), query_mask.to('cuda:1'), doc_tok.to('cuda:1'), doc_mask.to('cuda:1')).to('cuda:0')
        score_6 = self.bert_6(query_tok.to('cuda:1'), query_mask.to('cuda:1'), doc_tok.to('cuda:1'), doc_mask.to('cuda:1')).to('cuda:0')
        score = score_1 + score_2 + score_3 + score_4.to('cuda:0') + score_5.to('cuda:0') + score_6.to('cuda:0')

        return score

    def freeze_bert(self):
        print("freeze bert")
        self.bert_1.freeze_bert()
        self.bert_2.freeze_bert()
        self.bert_3.freeze_bert()
        self.bert_4.freeze_bert()
        self.bert_5.freeze_bert()
        self.bert_6.freeze_bert()

    def load_hexa(self, path1, path2, path3, path4, path5, path6):
        print("load hexa model")
        self.bert_1.load(path1)
        self.bert_2.load(path2)
        self.bert_3.load(path3)
        self.bert_4.load(path4)
        self.bert_5.load(path5)
        self.bert_6.load(path6)


class OctoBertRanker(MultiBertRanker):
    def __init__(self, sub_1, sub_2, sub_3, sub_4, sub_5, sub_6, sub_7, sub_8):
        super().__init__()
        self.bert_1 = sub_1.to('cuda:0')
        self.bert_2 = sub_2.to('cuda:0')
        self.bert_3 = sub_3.to('cuda:0')
        self.bert_4 = sub_4.to('cuda:0')
        self.bert_5 = sub_5.to('cuda:1')
        self.bert_6 = sub_6.to('cuda:1')
        self.bert_7 = sub_7.to('cuda:1')
        self.bert_8 = sub_8.to('cuda:1')

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        score_1 = self.bert_1(query_tok, query_mask, doc_tok, doc_mask)
        score_2 = self.bert_2(query_tok, query_mask, doc_tok, doc_mask)
        score_3 = self.bert_3(query_tok, query_mask, doc_tok, doc_mask)
        score_4 = self.bert_4(query_tok, query_mask, doc_tok, doc_mask)
        score_5 = self.bert_5(query_tok.to('cuda:1'), query_mask.to('cuda:1'), doc_tok.to('cuda:1'), doc_mask.to('cuda:1')).to('cuda:0')
        score_6 = self.bert_6(query_tok.to('cuda:1'), query_mask.to('cuda:1'), doc_tok.to('cuda:1'), doc_mask.to('cuda:1')).to('cuda:0')
        score_7 = self.bert_7(query_tok.to('cuda:1'), query_mask.to('cuda:1'), doc_tok.to('cuda:1'), doc_mask.to('cuda:1')).to('cuda:0')
        score_8 = self.bert_8(query_tok.to('cuda:1'), query_mask.to('cuda:1'), doc_tok.to('cuda:1'), doc_mask.to('cuda:1')).to('cuda:0')
        score = score_1 + score_2 + score_3 + score_4 + score_5.to('cuda:0') + score_6.to('cuda:0') + score_7.to('cuda:0') + score_8.to('cuda:0')

        #return score, score_1, score_2, score_3, score_4
        return score

    def freeze_bert(self):
        print("freeze bert")
        self.bert_1.freeze_bert()
        self.bert_2.freeze_bert()
        self.bert_3.freeze_bert()
        self.bert_4.freeze_bert()
        self.bert_5.freeze_bert()
        self.bert_6.freeze_bert()
        self.bert_7.freeze_bert()
        self.bert_8.freeze_bert()

    def load_octo(self, path1, path2, path3, path4, path5, path6, path7, path8):
        print("load octo model")
        self.bert_1.load(path1)
        self.bert_2.load(path2)
        self.bert_3.load(path3)
        self.bert_4.load(path4)
        self.bert_5.load(path5)
        self.bert_6.load(path6)
        self.bert_7.load(path7)
        self.bert_8.load(path8)


class CedrPacrrRanker(BertRanker):
    def __init__(self):
        super().__init__()
        QLEN = 20
        KMAX = 2
        NFILTERS = 32
        MINGRAM = 1
        MAXGRAM = 3
        self.simmat = modeling_util.SimmatModule()
        self.ngrams = torch.nn.ModuleList()
        self.rbf_bank = None
        for ng in range(MINGRAM, MAXGRAM+1):
            ng = modeling_util.PACRRConvMax2dModule(ng, NFILTERS, k=KMAX, channels=self.CHANNELS)
            self.ngrams.append(ng)
        qvalue_size = len(self.ngrams) * KMAX
        self.linear1 = torch.nn.Linear(self.BERT_SIZE + QLEN * qvalue_size, 32)
        self.linear2 = torch.nn.Linear(32, 32)
        self.linear3 = torch.nn.Linear(32, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        scores = [ng(simmat) for ng in self.ngrams]
        scores = torch.cat(scores, dim=2)
        scores = scores.reshape(scores.shape[0], scores.shape[1] * scores.shape[2])
        scores = torch.cat([scores, cls_reps[-1]], dim=1)
        rel = F.relu(self.linear1(scores))
        rel = F.relu(self.linear2(rel))
        rel = self.linear3(rel)
        return rel


class CedrKnrmRanker(BertRanker):
    def __init__(self):
        super().__init__()
        MUS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        SIGMAS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        self.bert_ranker = VanillaBertRanker()
        self.simmat = modeling_util.SimmatModule()
        self.kernels = modeling_util.KNRMRbfKernelBank(MUS, SIGMAS)
        self.combine = torch.nn.Linear(self.kernels.count() * self.CHANNELS + self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        kernels = self.kernels(simmat)
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        simmat = simmat.reshape(BATCH, 1, VIEWS, QLEN, DLEN) \
                       .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN) \
                       .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        result = kernels.sum(dim=3) # sum over document
        mask = (simmat.sum(dim=3) != 0.) # which query terms are not padding?
        result = torch.where(mask, (result + 1e-6).log(), mask.float())
        result = result.sum(dim=2) # sum over query terms
        result = torch.cat([result, cls_reps[-1]], dim=1)
        scores = self.combine(result) # linear combination over kernels
        return scores


class CedrDrmmRanker(BertRanker):
    def __init__(self):
        super().__init__()
        NBINS = 11
        HIDDEN = 5
        self.bert_ranker = VanillaBertRanker()
        self.simmat = modeling_util.SimmatModule()
        self.histogram = modeling_util.DRMMLogCountHistogram(NBINS)
        self.hidden_1 = torch.nn.Linear(NBINS * self.CHANNELS + self.BERT_SIZE, HIDDEN)
        self.hidden_2 = torch.nn.Linear(HIDDEN, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps, query_reps, doc_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        simmat = self.simmat(query_reps, doc_reps, query_tok, doc_tok)
        histogram = self.histogram(simmat, doc_tok, query_tok)
        BATCH, CHANNELS, QLEN, BINS = histogram.shape
        histogram = histogram.permute(0, 2, 3, 1)
        output = histogram.reshape(BATCH * QLEN, BINS * CHANNELS)
        # repeat cls representation for each query token
        cls_rep = cls_reps[-1].reshape(BATCH, 1, -1).expand(BATCH, QLEN, -1).reshape(BATCH * QLEN, -1)
        output = torch.cat([output, cls_rep], dim=1)
        term_scores = self.hidden_2(torch.relu(self.hidden_1(output))).reshape(BATCH, QLEN)
        return term_scores.sum(dim=1)

##
MODEL_MAP = {
    'vbert': VanillaBertRanker,
    'twinbert': TwinBertRanker,
    'twinrbert': TwinBertResRanker,
    'twinsbert': TwinBertSmallRanker,
    'colbert': ColBertRanker,
    'colvbert': ColBertVRanker,
    'knrmbert': KnrmBertRanker,
    'knrmvbert': KnrmVBertRanker,
    'drmmbert': DrmmBertRanker,
    'drmmvbert': DrmmVBertRanker,
    'duetbert': DuetBertRanker,
    'triobert': TrioBertRanker,
    'quadbert': QuadBertRanker,
    'octobert': OctoBertRanker,
    'cedr_pacrr': CedrPacrrRanker,
    'cedr_knrm': CedrKnrmRanker,
    'cedr_drmm': CedrDrmmRanker
}


