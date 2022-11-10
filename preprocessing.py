import pandas as pd


def combine(col1, col2):
    result = col1.replace("\n", " ")+" "+col2.replace("\n", " ")
    return result


def cutBlank(em):
    result = em.replace(" ","")
    return result


train = pd.read_csv('C:/projects/NLP/KoBERT/감성대화/감성대화말뭉치(최종데이터)_Training.csv')
valid = pd.read_csv('C:/projects/NLP/KoBERT/감성대화/감성대화말뭉치(최종데이터)_Validation.csv')

train['emotion'] = train.apply(lambda x: cutBlank(x['emotion']), axis=1)

mapping = {'기쁨': 0, '분노': 1, '중립': 2, '상처': 3, '슬픔': 4, '불안': 5, '당황': 6}
labeld_tr = train
labeld_tr['emotion'] = train.emotion.map(mapping)
labeld_va = valid
labeld_va['emotion'] = valid.emotion.map(mapping)

labeld_tr['sentences'] = labeld_tr.apply(lambda x: combine(x['sentence 1'], x['sentence 2']), axis=1)
labeld_va['sentences'] = labeld_va.apply(lambda x: combine(x['sentence 1'], x['sentence 2']), axis=1)

labeld_tr.drop(['sentence 1', 'sentence 2', 'sentence 3', 'sentence 4'], axis=1, inplace=True)
labeld_va.drop(['sentence 1', 'sentence 2', 'sentence 3', 'sentence 4'], axis=1, inplace=True)

labeld_tr.to_csv('C:/projects/NLP/KoBERT/감성대화/labeld_train.csv')
labeld_va.to_csv('C:/projects/NLP/KoBERT/감성대화/labeld_valid.csv')