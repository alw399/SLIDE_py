import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, LogisticRegression
from .knockoffs import Knockoffs


class Estimator():
    def __init__(self, model='linear', scaler='standard'):
        if model == 'linear':
            self.model = LinearRegression()
        elif model == 'logistic':
            self.model = LogisticRegression()
        else:
            raise ValueError(f"Invalid model: {model}")
        
        self.scaler = scaler
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def train_test_split(self, X, y, test_size=0.2, seed=1334):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed)
        return X_train, X_test, y_train, y_test

    def score(self, yhat, y):
        yhat = [1 if i >= 0.5 else 0 for i in yhat]
        if len(np.unique(y)) == 1:
            print('All same class')
            return None
        auc = roc_auc_score(y, yhat)
        return auc

    @staticmethod
    def scale_features(X, scaler, feature_range=(-1, 1)):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        if scaler == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif scaler == 'standard':
            scaler = StandardScaler()
        else:
            return X
            
        scaler.fit(X)
        return scaler.transform(X)

    def evaluate(self, X, y, n_iters=10, test_size=0.15):
        scores = [] 
        X = X.copy()

        X = self.scale_features(X, scaler=self.scaler)

        for iter in range(n_iters):
            X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size, seed=iter)
            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)
            scores.append(self.score(y_pred, y_test))
            
        return np.array(scores)
    
    @staticmethod
    def get_aucs(X, y, n_iters=10, test_size=0.2, scaler='standard'):
        estimator = Estimator(model='linear', scaler=scaler)
        return estimator.evaluate(X, y, n_iters, test_size)
    
class SLIDE_Estimator(Estimator):

    @staticmethod
    def score_performance(
            sig_LFs, sig_interacts, y, latent_factors, latent_factors_z2=None,
            n_iters=100, test_size=0.2, scaler='standard'):
        """
        Score the performance of the given latent factors relative to each other and to random
        selection of marginals and interactions.
        This visualization is similar to the control plot in original R code.

        s1: standalone LFs 
        s2: interacting LF terms (marginal * interacting)
        """

        if latent_factors_z2 is None: # This is only for SLIDE_PLM
            latent_factors_z2 = latent_factors

        scores = defaultdict(list)
        n = y.shape[0]
        
        s1 = latent_factors.loc[:, sig_LFs] # standalone LFs
        s2 = Knockoffs.get_interaction_terms(s1, latent_factors_z2.loc[:, sig_interacts]).reshape(n, -1)        # interaction LF terms
        s3 = np.concatenate([s1, s2], axis=1) if len(s2) > 0 else s1
        scores['s3'] = Estimator.get_aucs(s3, y, n_iters, test_size, scaler)

        n_marginals = s1.shape[1]
        n_interactions = s2.shape[1]
        sig_terms = np.concatenate([sig_LFs, sig_interacts], axis=0)
        non_marginal_terms = latent_factors.columns[~latent_factors.columns.isin(sig_LFs)]
        non_interaction_terms = latent_factors_z2.columns[~latent_factors_z2.columns.isin(sig_terms)]

        # To prevent out of bounds when doing np.random.choice
        n_interactions = min(n_interactions, len(non_interaction_terms))
        n_marginals = min(n_marginals, len(non_marginal_terms))

        for _ in range(n_iters):
            # Get random marginal terms
            s1_random = latent_factors.loc[:, np.random.choice(
                non_marginal_terms, 
                size=n_marginals, 
                replace=False
            )].values

            # Get random interaction terms
            s2_random = Knockoffs.get_interaction_terms(
                s1_random,
                latent_factors_z2.loc[:, np.random.choice(
                    non_interaction_terms,
                    size=n_interactions,
                    replace=False
                )]
            ).reshape(n, -1)

            # Combine random terms
            s3_random = (np.concatenate([s1_random, s2_random], axis=1) 
                        if len(s2_random) > 0 else s1_random)
            
            # Calculate scores for fully random selection
            scores['full_random'].append(
                Estimator.get_aucs(s3_random, y, 1, test_size, scaler)
            )

            # Calculate scores for partial random selection
            if n_interactions > 0:
                s2_real = Knockoffs.get_interaction_terms(
                    s1,
                    latent_factors_z2.loc[:, np.random.choice(
                        non_interaction_terms,
                        size=n_interactions,
                        replace=False
                    )]
                ).reshape(n, -1)

                s3_partial = (np.concatenate([s1, s2_real], axis=1) 
                            if len(s2_real) > 0 else s1)
                
                scores['partial_random'].append(
                    Estimator.get_aucs(s3_partial, y, 1, test_size, scaler)
                )

        if n_interactions > 0:
            scores['partial_random'] = np.array(scores['partial_random']).flatten()
        scores['full_random'] = np.array(scores['full_random']).flatten()

        return scores
