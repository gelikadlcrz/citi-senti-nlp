"""Optional category-only sentiment head for text-only inference.

This is a SEPARATE multi-label category-sentiment task, not the benchmark
annotated-target 3-class ACSA classifier. It handles opposing sentiment labels
for the same comment/category without silently discarding any annotations.

python -m models.linear_svm.category_sentiment
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from shared.data import WORKBOOK_DEFAULT, read_annotation_sheet
from shared.labels import ASPECT_NAMES, SENTIMENT_LABELS, SENTIMENT_TO_INDEX, normalize_aspect
from shared.tfidf import load_sentiment_vectorizer

MODEL_FILE = 'category_sentiment_model.joblib'
METRICS_FILE = 'category_sentiment_validation.json'

def make_category_input(model_text: str, aspect_code: str) -> str:
    code=normalize_aspect(aspect_code)
    if not str(model_text).strip():
        raise ValueError('Text must not be blank')
    return f'COMMENT: {model_text.strip()}  ASPECT_CATEGORY: {ASPECT_NAMES[code]}'

def category_examples(frame):
    texts=[]
    labels=[]
    for (_,code), group in frame.groupby(['comment_id','aspect_code'],sort=False):
        text=group['model_text'].iloc[0]
        texts.append(make_category_input(text,code))
        lab=np.zeros(len(SENTIMENT_LABELS),dtype=np.int8)
        for s in group['sentiment']:
            lab[SENTIMENT_TO_INDEX[s]]=1
        labels.append(lab)
    return texts, np.asarray(labels,dtype=np.int8)

def select_sentiments(margins: np.ndarray, threshold: float=0.0) -> np.ndarray:
    margins=np.asarray(margins,dtype=float)
    if margins.ndim!=2 or margins.shape[1]!=len(SENTIMENT_LABELS):
        raise ValueError('Expected three sentiment decision margins')
    pred=(margins>=threshold).astype(np.int8)
    missing=~pred.any(axis=1)
    if missing.any():
        pred[np.flatnonzero(missing),margins[missing].argmax(axis=1)]=1
    return pred

def train_category_sentiment(*,workbook=WORKBOOK_DEFAULT,
                             features='models/tfidf',artifacts='models/linear_svm/artifacts/tuned',
                             C=1.0,max_iter=5000,class_weight='balanced'):
    artifacts=Path(artifacts)
    if not (artifacts/'aspect_model.joblib').exists():
        raise FileNotFoundError('Run the main SVM tuner first')
    vec=load_sentiment_vectorizer(features)
    train_df=read_annotation_sheet(workbook,'Training')
    val_df=read_annotation_sheet(workbook,'Validation')
    t_train,y_train=category_examples(train_df)
    t_val,y_val=category_examples(val_df)
    model=OneVsRestClassifier(LinearSVC(C=C,max_iter=max_iter,class_weight=class_weight,random_state=42))
    model.fit(vec.transform(t_train),y_train)
    pred=select_sentiments(model.decision_function(vec.transform(t_val)))
    report={
       'task':'category_level_multilabel_sentiment',
       'is_primary_acsa_benchmark':False,
       'split':'Validation',
       'scope':'Gold-present category only, no target phrase supplied; not end-to-end',
       'note':'Some (comment, category) pairs have multiple sentiments; compare only against the same multi-label task.',
       'n_train':len(t_train),'n_val':len(t_val),
       'n_train_conflicting_categories':int((y_train.sum(axis=1)>1).sum()),
       'n_val_conflicting_categories':int((y_val.sum(axis=1)>1).sum()),
       'C':C,'class_weight':class_weight,'threshold':0.0,
       'macro_f1':float(f1_score(y_val,pred,average='macro',zero_division=0)),
       'micro_f1':float(f1_score(y_val,pred,average='micro',zero_division=0)),
       'macro_precision':float(precision_score(y_val,pred,average='macro',zero_division=0)),
       'macro_recall':float(recall_score(y_val,pred,average='macro',zero_division=0)),
       'exact_match_accuracy':float(accuracy_score(y_val,pred)),
    }
    joblib.dump(model,artifacts/MODEL_FILE)
    (artifacts/METRICS_FILE).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workbook',default=str(WORKBOOK_DEFAULT))
    p.add_argument('--features',default='models/tfidf')
    p.add_argument('--artifacts',default='models/linear_svm/artifacts/tuned')
    p.add_argument('--C',type=float,default=1.0)
    p.add_argument('--class-weight',choices=('none','balanced'),default='balanced')
    a=p.parse_args(argv)
    result=train_category_sentiment(workbook=a.workbook,features=a.features,artifacts=a.artifacts,
                                    C=a.C,class_weight=None if a.class_weight=='none' else a.class_weight)
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    main()
