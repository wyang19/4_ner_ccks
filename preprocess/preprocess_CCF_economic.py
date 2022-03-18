#coding:utf-8

import pandas as pd

import codecs
import re
import json
import sys
sys.path.append("CCF_ner/")  # 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块
from config import Config


print(sys.path)
"""
按照标点符号切割的预处理数据
"""
config = Config()
len_treshold = config.sequence_length - 2  #  每条数据的最大长度, 留下两个位置给[CLS]和[SEP]
data_dir = config.new_data_process_quarter_final
print(data_dir)

print(config.source_data_dir)
# 原始数据集
train_df = pd.read_csv(config.source_data_dir + 'Round2_train.csv', encoding='utf-8')
test_df = pd.read_csv(config.source_data_dir + 'Round2_Test.csv', encoding='utf-8')

# 找出所有的非中文、非英文和非数字符号
additional_chars = set()
for t in list(test_df.text) + list(train_df.text):
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', str(t)))

# 一些需要保留的符号
extra_chars = set("!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】")
print(extra_chars)
additional_chars = additional_chars.difference(extra_chars)

def stop_words(x):
    try:
        x = x.strip()
    except:
        return ''
    x = re.sub('{IMG:.?.?.?}', '', x)
    x = re.sub('<!--IMG_\d+-->', '', x)
    x = re.sub('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', x)  # 过滤网址
    x = re.sub('<a[^>]*>', '', x).replace("</a>", "")  # 过滤a标签
    x = re.sub('<P[^>]*>', '', x).replace("</P>", "")  # 过滤P标签
    x = re.sub('<strong[^>]*>', ',', x).replace("</strong>", "")  # 过滤strong标签
    x = re.sub('<br>', ',', x)  # 过滤br标签
    x = re.sub('www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', x).replace("()", "")  # 过滤www开头的网址
    x = re.sub('\s', '', x)   # 过滤不可见字符
    x = re.sub('Ⅴ', 'V', x)

    for wbad in additional_chars:
        x = x.replace(wbad, '')
    return x

train_df['text'] =  train_df['title'].fillna('') + train_df['text'].fillna('')
test_df['text'] =  test_df['title'].fillna('') + test_df['text'].fillna('')

# 清除噪声
train_df['text'] = train_df['text'].apply(stop_words)
test_df['text'] = test_df['text'].apply(stop_words)
train_df = train_df.fillna('')


"""
找出错误标签
"""
label_list = train_df['unknownEntities'].tolist()
text_list =  train_df['text'].tolist()
id_list =  train_df['id'].tolist()
false_get_id = []
false_get_label = []
for i, label in enumerate(label_list):
    text = text_list[i]
    idx = id_list[i]
    l_l = label.split(';')
    not_in = []
    for li in l_l:
        if li not in text:
            not_in.append(li)
    if len(not_in) > 0:

        false_get_id.append(idx)
        false_get_label.append(label)


"""
修复错误标签
"""

repair_id_label = ['大象健康科技有限公司;健康猫', '人人爱家金融', '速借贷;有信钱包', '速借贷;有信钱包', '速借贷;有信钱包',
'云讯通;云数贸;五行币;善心汇;LCF项目;云联惠;星火草原;云指商城;世界华人联合会;世界云联;WV梦幻之旅;维卡币;万福币;二元期权;云梦生活;恒星币;摩根币;网络黄金;1040阳光工程;中绿资本;赛比安;K币商城;五化联盟;国通通讯网络电话;EGD网络黄金;万达复利理财;MFC币理财;微转动力;神州互联商城;绿藤理财;绿色世界理财;宝微商城;中晋系;马克币;富迪;万通奇迹;港润信贷;CNC九星;世界云联;沃客生活;天音网络;莱汇币;盛大华天;惠卡世纪;开心理财网;贝格邦BGB;FIS数字金库;SF共享金融;DGC共享币;易赚宝;丰果游天下;天狮集团;薪金融;MGN积分宝;光彩币;亿加互助;GemCoin(珍宝币);老妈乐'


                  ]  # 对应id的修正实体
id_list = train_df['id'].tolist()
label_list = train_df['unknownEntities'].tolist()

for i, idx in enumerate(id_list):
    if idx in false_get_id :
        label_list[i] = repair_id_label[false_get_id.index(idx)]

# 修复过程中漏了几个标签，在这里补上
label_list[2409] = '金融科技（Fintech）'
label_list[2479] = '玖富钱包;玖富数科集团;玖富钱包APP'
label_list[3596] = '盈盈理财;乾包网;臻理财;蜗牛在线'
train_df['unknownEntities'] = label_list
train_df = train_df[~train_df['unknownEntities'].isnull()]  # 删除空标签
train_df.to_csv(data_dir + 'new_train_df.csv')

# 切分训练集，分成训练集和验证集，在这可以尝试五折切割
print('Train Set Size:', train_df.shape)
new_dev_df = train_df[4000:  ]
frames = [train_df[:2000], train_df[2001:4000]]
new_train_df = pd.concat(frames)  # 训练集
new_train_df = new_train_df.fillna('')
new_test_df = test_df[:]  # 测试集
new_test_df.to_csv(data_dir + 'new_test_df.csv', encoding='utf-8', index=False)


def _cut(sentence):
    """
    将一段文本切分成多个句子
    :param sentence:
    :return:
    """
    new_sentence = []
    sen = []
    for i in sentence:
        if i in ['。', '！', '？', '?'] and len(sen) != 0:
            sen.append(i)
            new_sentence.append("".join(sen))
            sen = []
            continue
        sen.append(i)

    if len(new_sentence) <= 1: # 一句话超过max_seq_length且没有句号的，用","分割，再长的不考虑了。
        new_sentence = []
        sen = []
        for i in sentence:
            if i.split(' ')[0] in ['，', ','] and len(sen) != 0:
                sen.append(i)
                new_sentence.append("".join(sen))
                sen = []
                continue
            sen.append(i)
    if len(sen) > 0:  # 若最后一句话无结尾标点，则加入这句话
        new_sentence.append("".join(sen))
    return new_sentence


# 数据切分
def cut_test_set(text_list):
    cut_text_list = []
    cut_index_list = []
    for text in text_list:

        temp_cut_text_list = []
        text_agg = ''
        if len(text) < len_treshold:
            temp_cut_text_list.append(text)
        else:
            sentence_list = _cut(text)  # 一条数据被切分成多句话
            for sentence in sentence_list:
                if len(text_agg) + len(sentence) < len_treshold:
                    text_agg += sentence
                else:
                    temp_cut_text_list.append(text_agg)
                    text_agg = sentence
            temp_cut_text_list.append(text_agg)  # 加上最后一个句子

        cut_index_list.append(len(temp_cut_text_list))
        cut_text_list += temp_cut_text_list

    return cut_text_list, cut_index_list


def cut_train_and_dev_set(text_list, label_list):
    cut_text_list = []
    cut_index_list = []
    cut_label_list = []
    for i, text in enumerate(text_list):
        if label_list[i] != '':
            text_label_list = label_list[i].split(';')  # 获取该条数据的实体列表
            temp_cut_text_list = []
            temp_cut_label_list = []
            text_agg = ''
            if len(text) < len_treshold:
                temp_cut_text_list.append(text)
                temp_cut_label_list.append(label_list[i])
            else:

                sentence_list = _cut(text)  # 一条数据被切分成多句话

                for sentence in sentence_list:
                    if len(text_agg) + len(sentence) < len_treshold:
                        text_agg += sentence
                    else:
                        new_label = []  # 新构成的句子的标签列表
                        for label in text_label_list:
                            if label in text_agg and label != '':
                                new_label.append(label)

                        if len(new_label) > 0:
                            temp_cut_text_list.append(text_agg)
                            temp_cut_label_list.append(";".join(new_label))

                        text_agg = sentence
                # 加回最后一个句子
                new_label = []
                for label in text_label_list:
                    if label in text_agg and label != '':
                        new_label.append(label)
                if len(new_label) > 0:
                    temp_cut_text_list.append(text_agg)
                    temp_cut_label_list.append(";".join(new_label))

            cut_index_list.append(len(temp_cut_text_list))
            cut_text_list += temp_cut_text_list
            cut_label_list += temp_cut_label_list

    return cut_text_list, cut_index_list, cut_label_list


train_text_list = new_train_df['text'].tolist()
train_label_list = new_train_df['unknownEntities'].tolist()
train_id_list = new_train_df['id'].tolist()

dev_text_list = new_dev_df['text'].tolist()
dev_label_list = new_dev_df['unknownEntities'].tolist()

test_text_list = new_test_df['text'].tolist()
test_id_list = new_test_df['id'].tolist()

train_cut_text_list, train_cut_index_list ,train_cut_label_list = cut_train_and_dev_set(train_text_list,  train_label_list)
dev_cut_text_list, dev_cut_index_list, dev_cut_label_list = cut_train_and_dev_set(dev_text_list, dev_label_list)

test_cut_text_list, cut_index_list = cut_test_set(test_text_list)

"""
测试切分是否正确
"""
flag = True

for i, text in enumerate(train_cut_text_list):
    label_list = train_cut_label_list[i].split(';')
    for li in label_list:
        if li not in text:
            print(i)
            print(li)
            print(text)
            flag = False
            print()
            break
        if li == '':

            print(li)
            print(text)
            flag = False
            print()
if flag:
    print("训练集切分正确！")
else:
    print("训练集切分错误！")


flag = True
for i, text in enumerate(dev_cut_text_list):
    label_list = dev_cut_label_list[i].split(';')
    for li in label_list:
        if li not in text:
            print(i)
            print(li)
            print(text)
            print()
            flag = False

if flag:
    print("验证集切分正确！")
else:
    print("验证集切分错误！")

# 保存切分索引
cut_index_dict = {'cut_index_list': cut_index_list}
with open(data_dir + 'cut_index_list.json', 'w') as f:
    json.dump(cut_index_dict, f, ensure_ascii=False)

dev_cut_index_dict = {'cut_index_list': dev_cut_index_list}
with open(data_dir + 'dev_cut_index_list.json', 'w') as f:
    json.dump(dev_cut_index_dict, f, ensure_ascii=False)


train_dict = {'text': train_cut_text_list, 'unknownEntities': train_cut_label_list}
train_df = pd.DataFrame(train_dict)

dev_dict = {'text': dev_cut_text_list, 'unknownEntities': dev_cut_label_list}
dev_df = pd.DataFrame(dev_dict)

test_dict = {'text': test_cut_text_list}
test_df = pd.DataFrame(test_dict)

print('训练集:', train_df.shape)
print('验证集:', dev_df.shape)
print('测试集:', test_df.shape)

# 构造训练集、验证集与测试集
with codecs.open(data_dir + 'train.txt', 'w', encoding='utf-8') as up:
    for row in train_df.iloc[:].itertuples():
        # print(row.unknownEntities)

        text_lbl = row.text
        entitys = str(row.unknownEntities).split(';')
        for entity in entitys:
            text_lbl = text_lbl.replace(entity, 'Ё' + (len(entity) - 1) * 'Ж')

        for c1, c2 in zip(row.text, text_lbl):
            if c2 == 'Ё':
                up.write('{0} {1}\n'.format(c1, 'B-ORG'))
            elif c2 == 'Ж':
                up.write('{0} {1}\n'.format(c1, 'I-ORG'))
            else:
                up.write('{0} {1}\n'.format(c1, 'O'))
        up.write('\n')

with codecs.open(data_dir + 'dev.txt', 'w', encoding='utf-8') as up:
    for row in dev_df.iloc[:].itertuples():
        # print(row.unknownEntities)
        text_lbl = row.text
        entitys = str(row.unknownEntities).split(';')
        for entity in entitys:
            text_lbl = text_lbl.replace(entity, 'Ё' + (len(entity) - 1) * 'Ж')

        for c1, c2 in zip(row.text, text_lbl):
            if c2 == 'Ё':
                up.write('{0} {1}\n'.format(c1, 'B-ORG'))
            elif c2 == 'Ж':
                up.write('{0} {1}\n'.format(c1, 'I-ORG'))
            else:
                up.write('{0} {1}\n'.format(c1, 'O'))

        up.write('\n')

with codecs.open(data_dir + 'test.txt', 'w', encoding='utf-8') as up:
    for row in test_df.iloc[:].itertuples():

        text_lbl = row.text
        for c1 in text_lbl:
            up.write('{0} {1}\n'.format(c1, 'O'))

        up.write('\n')
