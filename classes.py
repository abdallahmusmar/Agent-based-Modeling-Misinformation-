class NewsArticle:
    """This is a news article that has a title sentiment and a number of shares"""
    def __init__(
        self,
        news_article_id,
        source_id,
        source_preference,
        article_preference,
        sentiment,
        num_shares,
        fake,
        tick
    ):
        self.news_article_id = news_article_id
        self.source_id = source_id
        self.source_preference=source_preference
        self.article_preference = article_preference
        self.sentiment=sentiment #value between (-5,-1) and (1,5)
        self.num_shares = num_shares # any value >=0
        self.fake=fake #either real=0 or fake=1
        self.tick = tick
    
    def __repr__(self):
        return 'News Article ID: '+str(self.news_article_id)+'\nSource ID: '+str(self.source_id)+'\nSource Preference: '+str(self.source_preference)+'\nArticle Preference: '+str(self.article_preference)+'\nSentiment: '+str(self.sentiment)+'\nNumber of shares: '+str(self.num_shares)+'\nFake (True:1, False:0): '+str(self.fake)+'\nTick: '+str(self.tick)
    
    def __str__(self):
        return 'News Article ID: '+str(self.news_article_id)+'\nSource ID: '+str(self.source_id)+'\nSource Preference: '+str(self.source_preference)+'\nArticle Preference: '+str(self.article_preference)+'\nSentiment: '+str(self.sentiment)+'\nNumber of shares: '+str(self.num_shares)+'\nFake (True:1, False:0): '+str(self.fake)+'\nTick: '+str(self.tick)
        
class User():
    # THIS COULD BE THE USER
    
    #CONSTRUCTOR -- REPLACE WITH USER ATTRIBUTES
    def __init__(self, unique_id, news_spread_chance, preference, user_type, articles):
        self.unique_id = unique_id
        self.news_spread_chance = news_spread_chance
        self.preference = preference
        self.user_type = user_type
        self.articles = articles #set of article ids
        
    def __repr__(self):
        return 'User ID: '+str(self.unique_id)+'\nNews Spread Chance: '+str(self.news_spread_chance)+'\nPreference: '+str(self.preference)+'\nType: '+str(self.user_type)
    
    def __str__(self):
        return 'User ID: '+str(self.unique_id)+'\nNews Spread Chance: '+str(self.news_spread_chance)+'\nPreference: '+str(self.preference)+'\nType: '+str(self.user_type)

class NewsAgency(User):
    
    def __init__(self, unique_id, news_spread_chance, preference, user_type, articles, reliable):
        super(NewsAgency,self).__init__(unique_id, news_spread_chance, preference, user_type, articles)
        self.reliable = reliable
        
    def __repr__(self):
        return 'User ID: '+str(self.unique_id)+'\nNews Spread Chance: '+str(self.news_spread_chance)+'\nPreference: '+str(self.preference)+'\nType: '+str(self.user_type)+'\nReliable:'+str(self.reliable)
    
    def __str__(self):
        return 'User ID: '+str(self.unique_id)+'\nNews Spread Chance: '+str(self.news_spread_chance)+'\nPreference: '+str(self.preference)+'\nType: '+str(self.user_type)+'\nReliable:'+str(self.reliable)
            




