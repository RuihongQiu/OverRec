from recbole.model.abstract_recommender import SequentialRecommender


class OverRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(OverRec, self).__init__(config, dataset)
