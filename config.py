class Config:
    
    def __init__(self):
        
        self.embed_dense = True
        self.embed_dense_dim = 512  # 对BERT的Embedding降维
        self.warmup_proportion = 0.05
        self.use_bert = True
        self.keep_prob = 0.9
        self.relation_num = 10# 实体的种类

        self.decay_rate = 0.8
        self.decay_step = 2000
        self.num_checkpoints = 20 * 3
       # self.num_checkpoints = 3


        self.train_epoch = 30
        self.sequence_length = 128  # BERT的输入MAX_LEN

        self.learning_rate = 1e-4  # 下接结构的学习率
        self.embed_learning_rate = 5e-5  # BERT的微调学习率
        self.batch_size = 32
        
        self.model_type = 'bilstm' 
        self.lstm_dim = 256
        self.dropout = 0.5
        self.use_origin_bert = True


        #我的
        # medsave = 'E:\\modelsavepath'
        # BERT预训练模型的存放地址
        self.bert_file = './data/pengcheng01/pretrained_model/mc_bert_base/bert_model.ckpt'
        self.bert_config_file = ./data/pengcheng01/pretrained_model/mc_bert_base/bert_config.json'
        self.vocab_file = './data/pengcheng01/pretrained_model/mc_bert_base/vocab.txt'

        # self.bert_file = 'E:\\modelsavepathmodel\\runs_0\\1622903234/model_0.9057_0.9180-2244.ckpt'
        # self.bert_config_file = cur + '/data/pengcheng01/pretrained_model/RoBERTa-wwm-ext/bert_config.json'
        # self.vocab_file = cur + '/data/pengcheng01/pretrained_model/RoBERTa-wwm-ext/vocab.txt'

        # predict.py ensemble.py get_ensemble_final_result.py post_ensemble_final_result.py的结果路径
        self.continue_training = False
        self.ensemble_source_file = './data/wy/ccks/ensemble/source_file/'
        self.ensemble_result_file = './data/wy/ccks/ensemble/result_file/'

        # 存放的模型名称，用以预测
        self.checkpoint_path = "./data/wy/ccks/model/runs_7/1577502293/model_0.5630_0.6378-10305"  #

        self.model_dir = './data/wy/ccks/model'  # 模型存放地址
        self.new_data_process_quarter_final = './data/wy/ccks/clear_csv_data/data_process_quarter_final/'  # 数据预处理的结果路径
        self.source_data_dir = './data/wy/ccks/clear_csv_data/'  # 原始数据集

