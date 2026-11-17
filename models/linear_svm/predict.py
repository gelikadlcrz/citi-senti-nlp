"""Predict categories and, when a target phrase is provided, target sentiment.

python -m models.linear_svm.predict --text 'Mahal ang parking fee' --aspect-code A03 --aspect-text 'parking fee'

Plain --text does aspect detection only; there is no automatic target-span extractor.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
from models.linear_svm.thresholds import aspect_margins,apply_thresholds,load_thresholds
from models.linear_svm.category_sentiment import make_category_input, select_sentiments, MODEL_FILE
from shared.data import make_sentiment_input
from shared.labels import ASPECT_CODES,ASPECT_NAMES,SENTIMENT_LABELS,normalize_aspect
from shared.tfidf import load_aspect_vectorizer,load_sentiment_vectorizer

class LinearSVMPredictor:
    def __init__(self, *, artifacts='models/linear_svm/artifacts/tuned',features='models/tfidf'):
        root=Path(artifacts)
        self.aspect_model=joblib.load(root/'aspect_model.joblib')
        self.sentiment_model=joblib.load(root/'sentiment_model.joblib')
        self.aspect_vectorizer=load_aspect_vectorizer(features)
        self.sentiment_vectorizer=load_sentiment_vectorizer(features)
        self.cutoffs=load_thresholds(root)
        self.category_head_path=root/MODEL_FILE
        if self.aspect_model.n_features_in_!=len(self.aspect_vectorizer.vocabulary_):
            raise ValueError('Incompatible aspect model and vectorizer')
        if self.sentiment_model.n_features_in_!=len(self.sentiment_vectorizer.vocabulary_):
            raise ValueError('Incompatible sentiment model and vectorizer')

    def predict_aspects(self, model_text):
        if not isinstance(model_text,str) or not model_text.strip():
            raise ValueError('Non-empty text required')
        x=self.aspect_vectorizer.transform([model_text])
        flags=apply_thresholds(aspect_margins(self.aspect_model,x),self.cutoffs)[0]
        return [{'code':code,'name':ASPECT_NAMES[code]}
                for code,yes in zip(ASPECT_CODES,flags) if yes]

    def predict_comment(self,model_text):
        """Category-only end-to-end inference via optional separate category head."""
        if not self.category_head_path.exists():
            raise FileNotFoundError('Train optional category-only sentiment head first')
        model=joblib.load(self.category_head_path)
        if model.n_features_in_ != len(self.sentiment_vectorizer.vocabulary_):
            raise ValueError('Category head has wrong feature width')
        aspects=self.predict_aspects(model_text)
        if not aspects:
            return {'aspects':[], 'sentiment_scope':'category_level',
                    'note':'No category reached its decision margin cutoff.'}
        texts=[make_category_input(model_text,a['code']) for a in aspects]
        flags=select_sentiments(model.decision_function(self.sentiment_vectorizer.transform(texts)))
        return {'aspects':[
            {**a,'sentiments':[s for s,yes in zip(SENTIMENT_LABELS,row) if yes],
             'target_text':None} for a,row in zip(aspects,flags)],
            'sentiment_scope':'category_level',
            'note':'Category sentiments can be multi-label; target spans are not extracted.'}

    def predict_sentiment(self,model_text,aspect_code,aspect_text):
        x=self.sentiment_vectorizer.transform([make_sentiment_input(model_text,aspect_code,aspect_text)])
        return SENTIMENT_LABELS[int(self.sentiment_model.predict(x)[0])]

    def predict_targets(self,model_text,targets):
        return {'aspects':self.predict_aspects(model_text),
                'target_sentiments':[
                    {'aspect_code':normalize_aspect(t['aspect_code']),
                     'aspect_text':t['aspect_text'],
                     'sentiment':self.predict_sentiment(model_text,t['aspect_code'],t['aspect_text']),
                     'target_source':'supplied_by_caller'} for t in targets],
                'sentiment_scope':'target_conditioned'}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text',required=True)
    p.add_argument('--aspect-code')
    p.add_argument('--aspect-text')
    p.add_argument('--targets-json')
    p.add_argument('--category-only',action='store_true',help='Run optional category-only ABSA head')
    p.add_argument('--artifacts',default='models/linear_svm/artifacts/tuned')
    p.add_argument('--features',default='models/tfidf')
    a=p.parse_args(argv)
    if (a.aspect_code is None)!=(a.aspect_text is None):
        p.error('--aspect-code and --aspect-text must be supplied together')
    if a.category_only and (a.targets_json or a.aspect_code):
        p.error('--category-only cannot be combined with explicit target input')
    if a.targets_json and a.aspect_code:
        p.error('Use --targets-json OR --aspect-code/--aspect-text')
    svm=LinearSVMPredictor(artifacts=a.artifacts,features=a.features)
    if a.category_only:
        result=svm.predict_comment(a.text)
    elif a.targets_json:
        result=svm.predict_targets(a.text,json.loads(a.targets_json))
    elif a.aspect_code:
        result=svm.predict_targets(a.text,[{'aspect_code':a.aspect_code,'aspect_text':a.aspect_text}])
    else:
        result={'aspects':svm.predict_aspects(a.text),'sentiment_scope':'not_requested',
                'note':'A target expression is required for this benchmark sentiment classifier.'}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
if __name__=='__main__':
    main()
