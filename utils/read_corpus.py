import os

sentence_begin='<s>'
sentence_end  ='</s>'

def read_corpus(file_path, stage = 'train', corpus_type = 'gloss'):
    filenames = []
    for i in os.walk(file_path):
        filenames = i[2]
    data = []
    filename=''
    for i in filenames:
        if stage in i and corpus_type in i:
            filename=i
            break
    for line in open(file_path + filename):
        sent = line.strip().split(' ')
        sent = [sentence_begin] + sent +[sentence_end]
        data.append(sent)
    return data

if __name__ == "__main__":
    file_path = "F:/学习资料/基金项目-手语/nslt-master/Data/"
    stage = "train"
    corpus_type  = 'gloss'
    read_corpus(file_path,stage,corpus_type)