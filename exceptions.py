"""
Exceptions and warnings particular to hierachical multi-classification.
"""

__all__ = ["NoSamplesForStageWarning", "StageNotFitWarning", "ClassifierNotFitError"]


class NoSamplesForStageWarning(UserWarning):
    """Warning used to notify that classification stage has no eligible samples.
    This warning happens when no samples in the input set (the design matrix) are eligible for
    classification at the current stage. This can happen during training or prediction if no
    samples are in, or descend from, the class in the class hierarchy corresponding to the stage.
    """


class StageNotFitWarning(UserWarning):
    """Warning used to notify that no estimator was fit for the classification stage.
    This warning happens when samples are eligible for prediction at a classification stage that
    was not fit when the hierachical classifier was fit. This can happen if the training set used
    to fit the hierachical classifier had no samples in, or descending from, the class in the
    class hierarchy corresponding to the stage.
    """


class ClassifierNotFitError(ValueError):
    """Warning used to notify that no estimators were fit for the hierachical classifier.
    This warning happens when the hierachical classifier is exploited without first being fit.
    """
