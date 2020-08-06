from sklearn.metrics import cohen_kappa_score, r2_score, mean_squared_error, mean_absolute_error, \
                            mean_squared_log_error, accuracy_score, roc_auc_score, matthews_corrcoef, \
                            average_precision_score, mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score, \
                            precision_score, recall_score, f1_score, jaccard_score

from sklearn.utils import check_array
import numpy as np
from scipy.stats import spearmanr

try:
    from pytorch_lightning.metrics.metric import NumpyMetric
    class PLMetric(NumpyMetric):

        def __init__(self, metric_name, convert=None, reshape=False):
            super(PLMetric, self).__init__(metric_name)
            self.scorer = get_scorer(metric_name)
            self.convert = convert
            self.reshape = reshape

        def forward(self,*args):
            if len(args) == 2:
                x = args[0]
                y = args[1]

                x = np.array(x)
                y = np.array(y)

                if self.convert == 'round':
                    x = np.round(x)
                    y = np.round(y)
                if self.convert == 'max':
                    #x = np.argmax(x, -1)
                    y = np.argmax(y, -1)
                if self.convert == 'min':
                    #x = np.argmin(x, -1)
                    y = np.argmin(y, -1)

                if self.reshape:
                    x = x.reshape(-1,1)
                    y = y.reshape(-1,1)

                return self.scorer(x,y)

            elif len(args) == 4:
                x1 = args[0]
                x2 = args[1]
                y1 = args[2]
                y2 = args[3]

                x1 = np.array(x1)
                x2 = np.array(x2)
                y1 = np.array(y1)
                y2 = np.array(y2)

                if self.convert == 'round':
                    x1 = np.round(x1)
                    x2 = np.round(x2)
                    y1 = np.round(y1)
                    y2 = np.round(y2)
                if self.convert == 'max':
                    #x = np.argmax(x, -1)
                    y1 = np.argmax(y1, -1)
                    y2 = np.argmax(y2, -1)
                if self.convert == 'min':
                    #x = np.argmin(x, -1)
                    y1 = np.argmin(y1, -1)
                    y2 = np.argmin(y2, -1)

                if self.reshape:
                    x1 = x1.reshape(-1,1)
                    x2 = x2.reshape(-1,1)
                    y1 = y1.reshape(-1,1)
                    y2 = y2.reshape(-1,1)

                return self.scorer(x1,x2,y1,y2)
            
            else:
                raise ValueError("arguments not understood")


except:
    pass

class SKMetric:
    def __init__(self, metric_name, convert=None, reshape=False):
        self.scorer = get_scorer(metric_name)
        self.convert = convert
        self.reshape = reshape

    def __call__(self, *args):
        if len(args) == 2:
            x = args[0]
            y = args[1]

            x = np.array(x)
            y = np.array(y)

            if self.convert == 'round':
                x = np.round(x)
                y = np.round(y)
            if self.convert == 'max':
                #x = np.argmax(x, -1)
                y = np.argmax(y, -1)
            if self.convert == 'min':
                #x = np.argmin(x, -1)
                y = np.argmin(y, -1)

            if self.reshape:
                x = x.reshape(-1,1)
                y = y.reshape(-1,1)

            return self.scorer(x,y)

        elif len(args) == 4:
            x1 = args[0]
            x2 = args[1]
            y1 = args[2]
            y2 = args[3]

            x1 = np.array(x1)
            x2 = np.array(x2)
            y1 = np.array(y1)
            y2 = np.array(y2)

            if self.convert == 'round':
                x1 = np.round(x1)
                x2 = np.round(x2)
                y1 = np.round(y1)
                y2 = np.round(y2)
            if self.convert == 'max':
                #x = np.argmax(x, -1)
                y1 = np.argmax(y1, -1)
                y2 = np.argmax(y2, -1)
            if self.convert == 'min':
                #x = np.argmin(x, -1)
                y1 = np.argmin(y1, -1)
                y2 = np.argmin(y2, -1)

            if self.reshape:
                x1 = x1.reshape(-1,1)
                x2 = x2.reshape(-1,1)
                y1 = y1.reshape(-1,1)
                y2 = y2.reshape(-1,1)

            return self.scorer(x1,x2,y1,y2)
        
        else:
            raise ValueError("arguments not understood")

def jaccard(l1,l2):
    return len(set(l1).intersection(l2))*1.0/(len(l1) + len(l2) - len(set(l1).intersection(l2)))

def qa_jaccard_score(target_start, target_end, output_start, output_end):
    if type(target_start) == np.ndarray:
        target_start = target_start.flatten()
        target_end = target_end.flatten() 
        output_start = output_start.flatten() 
        output_end = output_end.flatten()

        return np.mean(np.array([jaccard(np.arange(target_start[i],target_end[i]+1),np.arange(output_start[i], output_end[i]+1)) for i in range(target_start.shape[0])]))
    
    else:

        return jaccard(np.arange(target_start,target_end+1),np.arange(output_start, output_end+1))

class ranking_precision_score:
    def __init__(self, k=10):
        self.k = k

    def __call__(self, y_true, y_score):
        k = self.k
        """Precision at rank k
        Parameters
        ----------
        y_true : array-like, shape = [n_samples]
            Ground truth (true relevance labels).
        y_score : array-like, shape = [n_samples]
            Predicted scores.
        k : int
            Rank.
        Returns
        -------
        precision @k : float
        """
        unique_y = np.unique(y_true)

        if len(unique_y) > 2:
            raise ValueError("Only supported for two relevance levels.")

        pos_label = unique_y[1]
        n_pos = np.sum(y_true == pos_label)

        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])
        n_relevant = np.sum(y_true == pos_label)

        # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
        return float(n_relevant) / min(n_pos, k)

def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


class ndcg_score:
    def __init__(self,k=10,gains='exponential'):
        self.k = k
        self.gains = gains

    def __call__(self, y_true, y_score):
        k = self.k
        gains = self.gains
        """Normalized discounted cumulative gain (NDCG) at rank k
        Parameters
        ----------
        y_true : array-like, shape = [n_samples]
            Ground truth (true relevance labels).
        y_score : array-like, shape = [n_samples]
            Predicted scores.
        k : int
            Rank.
        gains : str
            Whether gains should be "exponential" (default) or "linear".
        Returns
        -------
        NDCG @k : float
        """
        best = dcg_score(y_true, y_true, k, gains)
        actual = dcg_score(y_true, y_score, k, gains)
        return actual / best


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = check_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def spearmans_rho(y_true, y_pred, axis=0):
    """
        Calculates the Spearman's Rho Correlation between ground truth labels and predictions
    """
    return spearmanr(y_true, y_pred, axis=axis).correlation


def sklearn_qwk(y_true, y_pred) -> np.float64:
    """
    Function for measuring Quadratic Weighted Kappa with scikit-learn

    :param y_true: The ground truth labels
    :param y_pred: The predicted labels

    :return The Quadratic Weighted Kappa Score (QWK)
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def get_scorer(scoring):
    """Get a scorer from string.
    Read more in the :ref:`User Guide <scoring_parameter>`.
    Parameters
    ----------
    scoring : str | callable
        scoring method as string. If callable it is returned as is.
    Returns
    -------
    scorer : callable
        The scorer.
    """
    if isinstance(scoring, str):
        try:
            scorer = SCORERS[scoring]
        except KeyError:
            raise ValueError('{} is not a valid scoring value. '
                             'Use sorted({}) '
                             'to get valid options.'.format(scoring,SCORERS.keys()))
    else:
        scorer = scoring
    return scorer

class _Scorer:
    def __init__(self, score_func, **kwargs):
        self._score_func = score_func
        self._kwargs = kwargs

    def __call__(self, y_true, y_pred, sample_weight=None):

        if sample_weight is not None:
            return self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,**self._kwargs)
        else:
            return self._score_func(y_true, y_pred, **self._kwargs)

# Standard regression scores
r2_scorer = _Scorer(r2_score)
mse_scorer = _Scorer(mean_squared_error)
mae_scorer = _Scorer(mean_absolute_error)
msle_scorer = _Scorer(mean_squared_log_error)
mape_scorer = _Scorer(mean_absolute_percentage_error)

# Classification scores
accuracy_scorer = _Scorer(accuracy_score)
roc_auc_scorer = _Scorer(roc_auc_score)
spearman_scorer = _Scorer(spearmans_rho)
qwk_scorer = _Scorer(sklearn_qwk)
mathew_corr_scorer = _Scorer(matthews_corrcoef)
average_precision_scorer = _Scorer(average_precision_score)

# clustering scores
mutual_info_scorer = _Scorer(mutual_info_score)
adjusted_mutual_info_scorer = _Scorer(adjusted_mutual_info_score)
normalized_mutual_info_scorer = _Scorer(normalized_mutual_info_score)

SCORERS = dict(r2_scorer=r2_scorer,
               mse_scorer=mse_scorer,
               mae_scorer=mae_scorer,
               msle_scorer=msle_scorer,
               mape_scorer=mape_scorer,
               accuracy_scorer=accuracy_scorer,
               roc_auc_scorer=roc_auc_scorer,
               spearman_scorer=spearman_scorer,
               qwk_scorer=qwk_scorer,
               mathew_corr_scorer=mathew_corr_scorer,
               average_precision_scorer=average_precision_scorer,
               mutual_info_scorer=mutual_info_scorer,
               adjusted_mutual_info_scorer=adjusted_mutual_info_scorer,
               normalized_mutual_info_scorer=normalized_mutual_info_scorer,
               ranking_precision_scorer=ranking_precision_score(k=50),
               ndcg_scorer=ndcg_score(k=50),
               qa_jaccard=qa_jaccard_score
               )

for name, metric in [('precision', precision_score),
                     ('recall', recall_score), ('f1', f1_score),
                     ('jaccard', jaccard_score)]:
    SCORERS[name] = _Scorer(metric, average='binary')
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        SCORERS[qualified_name] = _Scorer(metric, pos_label=None,
                                              average=average)
