class TweetFormer:

    def __init__(self,data):
        self.n_sent = 1
        self.data = data
        self.empty = False

        agg_func = lambda s: [(w,p,t) for w,p,t in zip(s['word'].values.tolist(), s['pos_tag'].values.tolist(), s['ner_tag'].values.tolist())]

        self.grouped = self.data.groupby('tweet_id').apply(agg_func)
        self.tweets = [s for s in self.grouped]


    def gen_next(self):
        try:
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None