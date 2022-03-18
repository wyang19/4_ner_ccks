# -*- coding:utf-8 -*-
import sys
import codecs
import os
import glob
from sklearn.model_selection import train_test_split


label_dict = {
    '检查和检验': 'CHECK',
    '症状和体征': 'SIGNS',
    '疾病和诊断': 'DISEASE',
    '治疗': 'TREATMENT',
    '身体部位': 'BODY'
    }


text_length = 250


file_list = glob.glob('./mydata/*.txt')


train_filelist, val_filelist = train_test_split(file_list, test_size=0.1, random_state=666)
print(train_filelist)
print(val_filelist)

all_filelist = file_list

def _cut(sentence):

    new_sentence = []
    sen = []
    for i in sentence:
        if i in ['。', '！', '？', '?'] and len(sen) != 0:
            sen.append(i)
            new_sentence.append("".join(sen))
            sen = []
            continue
        sen.append(i)

    if len(new_sentence) <= 1:
        new_sentence = []
        sen = []
        for i in sentence:
            if i.split(' ')[0] in ['，', ','] and len(sen) != 0:
                sen.append(i)
                new_sentence.append("".join(sen))
                sen = []
                continue
            sen.append(i)
    if len(sen) > 0:
        new_sentence.append("".join(sen))
    return new_sentence


def cut_test_set(text_list, len_treshold):
    cut_text_list = []
    cut_index_list = []
    for text in text_list:

        temp_cut_text_list = []
        text_agg = ''
        if len(text) < len_treshold:
            temp_cut_text_list.append(text)
        else:
            sentence_list = _cut(text)
            for sentence in sentence_list:
                if len(text_agg) + len(sentence) < len_treshold:
                    text_agg += sentence
                else:
                    temp_cut_text_list.append(text_agg)
                    text_agg = sentence
            temp_cut_text_list.append(text_agg)

        cut_index_list.append(len(temp_cut_text_list))
        cut_text_list += temp_cut_text_list

    return cut_text_list, cut_index_list

def from_ann2dic(r_ann_path, r_txt_path, w_path, w_file):

    q_dic = {}
    with codecs.open(r_ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split('\t')
            # line = line.strip("\n\r")
            # line_arr = line.split('\t')
            # entityinfo = line_arr[1]
            # entityinfo = entityinfo.split(' ')
            cls = line[3]
            start_index = int(line[1])
            end_index = int(line[2])
            # length = end_index - start_index
            # length = length
            for r in range(start_index,end_index+1):
                if r == start_index:
                    q_dic[start_index] = ("%s-B" % label_dict[cls])
                else:
                    q_dic[r] = ("%s-I" % label_dict[cls])

    with codecs.open(r_txt_path, "r", encoding="utf-8") as f:
        content_str = f.read()

    cut_text_list, cut_index_list = cut_test_set([content_str], text_length)

    i = 0
    for idx, line in enumerate(cut_text_list):
        w_path_ = "%s/%s-%s-new.txt" % (w_path, w_file, idx)
        with codecs.open(w_path_, "w", encoding="utf-8") as w:
            for str_ in line:
                if str_ is " " or str_ == "" or str_ == "\n" or str_ == "\r":
                    pass
                else:
                    if i in q_dic:
                        tag = q_dic[i]
                    else:
                        tag = "O"
                    w.write('%s\t%s\n' % (str_, tag))
                i += 1
            w.write('%s\n' % "END\tO")

def deal_train_data():

    data_dir = './mydata/'
    for file in train_filelist:
        filepath = os.path.join(data_dir, file)
        if 'original' not in filepath:
            continue
        label_filepath = filepath.replace('.txtoriginal', '')
        # if file.find(".ann") == -1 and file.find(".txt") == -1:
        #     continue
        file_name = label_filepath.split('/')[-1].split('.')[0].split('\\')[1]
        r_ann_path = os.path.join(data_dir, "%s.txt" % file_name)
        r_txt_path = os.path.join(data_dir, "%s.txtoriginal.txt" % file_name)
        w_path = './train_new/'
        w_file = file_name
        from_ann2dic(r_ann_path, r_txt_path, w_path, w_file)


    w_path = "./train/train.txt"
    for file in os.listdir('./train_new/'):
        path = os.path.join("./train_new", file)
        if not file.endswith(".txt"):
            continue
        q_list = []
        print("开始读取文件:%s" % file)
        with codecs.open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            line = line.strip("\n\r")
            while line != "END\tO":
                q_list.append(line)
                line = f.readline()
                line = line.strip("\n\r")
        print("开始写入文本%s" % w_path)
        with codecs.open(w_path, "a", encoding="utf-8") as f:
            for item in q_list:
                if item.__contains__('\ufeff1'):
                    print("===============")
                f.write('%s\n' % item)
            f.write('\n')
        f.close()

def deal_all_data():

    data_dir = './mydata/'
    for file in all_filelist:
        filepath = os.path.join(data_dir, file)
        if 'original' not in filepath:
            continue
        label_filepath = filepath.replace('.txtoriginal', '')
        # if file.find(".ann") == -1 and file.find(".txt") == -1:
        #     continue
        file_name = label_filepath.split('/')[-1].split('.')[0].split('\\')[1]
        r_ann_path = os.path.join(data_dir, "%s.txt" % file_name)
        r_txt_path = os.path.join(data_dir, "%s.txtoriginal.txt" % file_name)
        # if file.find(".ann") == -1 and file.find(".txt") == -1:
        #     continue
        # file_name = file.split('/')[-1].split('.')[0].split('\\')[1]
        # r_ann_path = os.path.join(data_dir, "%s.ann" % file_name)
        # r_txt_path = os.path.join(data_dir, "%s.txt" % file_name)
        w_path = './all_data_new/'
        w_file = file_name
        from_ann2dic(r_ann_path, r_txt_path, w_path,w_file)


    w_path = "./all_data/all.txt"
    for file in os.listdir('./all_data_new/'):
        path = os.path.join("./all_data_new", file)
        if not file.endswith(".txt"):
            continue
        q_list = []
        print("开始读取文件:%s" % file)
        with codecs.open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            line = line.strip("\n\r")
            while line != "END\tO":
                q_list.append(line)
                line = f.readline()
                line = line.strip("\n\r")
        print("开始写入文本%s" % w_path)
        with codecs.open(w_path, "a", encoding="utf-8") as f:
            for item in q_list:
                if item.__contains__('\ufeff1'):
                    print("===============")
                f.write('%s\n' % item)
            f.write('\n')
        f.close()

def deal_val_data():

    import os
    import codecs
    data_dir = './mydata/'
    for file in val_filelist:
        filepath = os.path.join(data_dir, file)
        if 'original' not in filepath:
            continue
        label_filepath = filepath.replace('.txtoriginal', '')
        # if file.find(".ann") == -1 and file.find(".txt") == -1:
        #     continue
        file_name = label_filepath.split('/')[-1].split('.')[0].split('\\')[1]
        r_ann_path = os.path.join(data_dir, "%s.txt" % file_name)
        r_txt_path = os.path.join(data_dir, "%s.txtoriginal.txt" % file_name)
        # if file.find(".ann") == -1 and file.find(".txt") == -1:
        #     continue
        # file_name = file.split('/')[-1].split('.')[0].split('\\')[1]
        # r_ann_path = os.path.join(data_dir, "%s.ann" % file_name)
        # r_txt_path = os.path.join(data_dir, "%s.txt" % file_name)
        w_path = './val_new/'
        w_file = file_name
        from_ann2dic(r_ann_path, r_txt_path, w_path,w_file)


    w_path = "./val/val.txt"
    for file in os.listdir('./val_new/'):
        path = os.path.join("./val_new", file)
        if not file.endswith(".txt"):
            continue
        q_list = []

        with codecs.open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            line = line.strip("\n\r")
            while line != "END\tO":
                q_list.append(line)
                line = f.readline()
                line = line.strip("\n\r")

        with codecs.open(w_path, "a", encoding="utf-8") as f:
            for item in q_list:
                if item.__contains__('\ufeff1'):
                    print("===============")
                f.write('%s\n' % item)
            f.write('\n')
        f.close()

def copy_val_data():

    os.system(f"mkdir  ./val_data/")

    for file in val_filelist:

        file_name = file.split('/')[-1].split('.')[0].split('\\')[1]
        r_ann_path = os.path.join("./mydata", "%s.txt" % file_name)
        os.system("cp %s %s" % (file, "./val_data/"))
        os.system("cp %s %s" % (r_ann_path, "./val_data/"))
        print(file)

if __name__ == '__main__':
    deal_train_data()
    deal_val_data()
    deal_all_data()
    copy_val_data()





