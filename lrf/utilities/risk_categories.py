from lrf.utilities import utility
import os
from collections import defaultdict
import json

## We re hardcoding the Keywords for now for each risk category

def main():
    lrf_path = os.getcwd()
    proj_path = os.path.abspath(os.path.join(lrf_path, os.pardir))
    ref_data_path = os.path.join(proj_path, 'reference_data')
    risk_cat_file = os.path.join(ref_data_path, 'risk_category_file.json')

    glove_file = os.path.join(ref_data_path, 'glove_data/glove.twitter.27B.200d.txt')

    category_keywords = defaultdict(dict)

    ## Tag:	achievement
    achievement_pos = ['acne', 'school', 'cancer', 'ai', 'india', 'lazy', 'watches', 'hair', 'growth', 'tumours',
                       'surgery', 'technology', 'happiness', 'industries', 'tumour', 'dna', 'employment', 'nabin',
                       'research', 'access', 'authorized', 'samples', 'scientists', 'test', 'prof', 'method',
                       'satellite', 'used', 'jobs', 'electric', 'foreign domestic', 'robots', 'zuckerberg', 'facebook',
                       'blood', 'device', 'using', 'lymphoma', 'data', 'security', 'training', 'cells', 'robot', 'page',
                       'patients', 'tissue', 'team', 'bots', 'university', 'researchers']
    achievement_neg = ['beijing', 'trump', 'family', 'photo', 'north korea', 'woman', 'samsung', 'sexual', 'city',
                       'united', 'police', 'music', 'nuclear', 'attack', 'state', 'health', 'statement', 'party',
                       'korean', 'man', 'war', 'told', 'korea', 'north', 'government', 'million', 'infection',
                       'missile', 'islamic', 'incident', 'china', 'authorities', 'news', 'killed', 'died', 'zika',
                       'case', 'states', 'several', 'last', 'products', 'ministry', 'official', 'well', 'us', 'air',
                       'officials', 'dengue', 'mr', 'south']

    ## Tag:	economics
    economics_pos = ['financial', 'investors', 'cuts', 'money', 'global', 'tax', 'per cent', 'cryptocurrency',
                     'currency', 'growth', 'tencent', 'per', 'year', 'investment', 'market', 'retirement', 'buffett',
                     'bitcoin', 'cent', 'uob', 'projects', 'income', 'units', 'medifund', 'markets', 'patricia',
                     'economy', 'firm', 'currencies', 'finance', 'chinas', 'funds', 'million', 'fund', 'inflation',
                     'wechat', 'demand', 'prices', 'hedge', 'trade', 'data', 'bank', 'billion', 'companies', 'dollar',
                     'asean', 'economic', 'mr', 'banks', 'quarter']
    economics_neg = ['heart', 'apple', 'cancer', 'people', 'national', 'among', 'one', 'north korea', 'woman',
                     'election', 'talks', 'death', 'police', 'chinese', 'nuclear', 'hospital', 'research', 'attack',
                     'camera', 'health', 'test', 'zika', 'daily', 'body', 'korea', 'food', 'men', 'missile', 'water',
                     'virus', 'blood', 'found', 'report', 'court', 'cases', 'dr', 'killed', 'man', 'case', 'training',
                     'samsung', 'study', 'devices', 'air', 'patients', 'rohingya', 'dengue', 'military', 'security',
                     'researchers']

    ## Tag:	politics
    politics_pos = ['communist party', 'xi', 'trump', 'campaign', 'senate', 'madrid', 'mnangagwa', 'elections', 'eu',
                    'donald', 'election', 'budget', 'parties', 'vote', 'politics', 'dementia', 'leaders', 'would',
                    'franken', 'political', 'state', 'ruling', 'policy', 'party', 'communist', 'leader', 'mugabe',
                    'opposition', 'parliament', 'general', 'congress', 'government', 'prime minister', 'resign',
                    'minister', 'advertisement', 'moore', 'president', 'republican', 'former', 'presidential', 'prime',
                    'tobacco', 'coalition', 'corruption', 'trumps', 'mr', 'senator', 'democratic', 'decision']
    politics_neg = ['apple', 'cancer', 'back', 'malaysia', 'cent', 'still', 'market', 'risk', 'fire', 'police',
                    'myanmar', 'per cent', 'per', 'research', 'students', 'camera', '5', 'treatment', 'ebola',
                    'products', 'zika', 'exercise', 'iphone', 'used', 'users', 'get', 'food', 'company', 'men', 'water',
                    'virus', 'brain', 'blood', 'dr', 'man', 'like', 'samsung', 'mobile', 'study', 'violence', 'disease',
                    'network', 'patients', 'rohingya', 'games', 'road', 'found', 'residents', 'medical', 'researchers']

    ## Tag:	not_applicable
    na_pos = ['madam', 'set', 'says', 'features', 'battery', 'photo', 'app', 'sales', 'streaming', 'google', 'design',
              'devices', 'sports', 'market', 'apple', 'lg', 'service', 'tv', 'museum', 'apps', 'best', 'zoo', 'halimah',
              'camera', 'music', 'iphone', '5', 'electronics', 'company', 'store', '4', 'customers', 'smartphone',
              'users', 'screen', 'watch', 'game', 'new', 'galaxy', 'android', 'billion', 'like', 'samsung', 'mobile',
              'experience', 'display', 'games', 'video', 'microsoft', 'first']
    na_neg = ['xi', 'cancer', 'people', 'states', 'officials', 'eu', 'said', 'police', 'hospital', 'agency', 'attack',
              'state', 'reuters', 'health', 'treatment', 'party', 'zika', 'told', 'korea', 'north', 'risk',
              'government', 'fire', 'infection', 'reported', 'virus', 'united', 'minister', 'advertisement', 'child',
              'cases', 'law', 'dr', 'killed', 'women', 'north korea', 'attacks', 'economic', 'medical', 'official',
              'disease', 'drug', 'patients', 'train', 'dengue', 'found', 'security', 'study', 'researchers', 'blood']

    ## Tag:	skip
    skip_pos = ['swipe', 'consider', 'programmable chips', 'battery', 'relegated', 'vice versa', 'watches', 'martial',
                'clubs', 'grappling', 'arts', 'probability', 'mma', 'bipolar disorder', 'hybrid', 'swipe wrist',
                'teams compete', 'bipolar', 'versa', 'programmable', 'football association singapore', 'martial arts',
                'programmable chips watches', 'division', 'championship', 'plans', 'payments swipe wrist', 'season',
                'smartwatch', 'victory', 'chips', 'swiss', 'wrist', 'sian', 'swatch', 'one championship', 'cage',
                'running clubs', 'sleague', 'classifications', 'career', 'told swiss', 'singapore going', 'players',
                'chips watches', 'imran', 'professional', 'disorder', 'payments swipe', 'football association']
    skip_neg = ['trump', 'chinese', 'people', 'national', 'two', 'north korea', 'added', 'cent', 'year', 'children',
                'even', 'said', 'police', 'cancer', 'would', 'hospital', 'singapore', 'per cent', 'per', 'also',
                'health', 'take', 'online', 'new', 'public', 'korea', 'north', 'around', 'government', 'may', 'million',
                '000', 'china', 'minister', 'advertisement', 'president', 'cases', 'years', 'dr', 'women', 'last',
                'many', 'united', 'us', 'patients', 'three', 'time', 'security', 'south', 'first']

    ## Tag:	health
    health_pos = ['heart', 'help', 'cancer', 'people', 'flu', 'mers', 'condition', 'brain', 'sleep', 'diabetes',
                  'patient', 'diseases', 'drugs', 'women', 'eye', 'doctor', 'hospital', 'fat', 'symptoms', 'may',
                  'health', 'treatment', 'ebola', 'outbreak', 'care', 'pain', 'risk', 'hiv', 'get', 'food',
                  'ecigarettes', 'infection', 'virus', 'blood', 'doctors', 'smoking', 'cases', 'dr', 'vaccine', 'zika',
                  'products', 'infected', 'study', 'infections', 'disease', 'drug', 'patients', 'dengue', 'medical',
                  'researchers']
    health_neg = ['trump', 'chinese', 'app', 'tax', 'north korea', 'video', 'election', 'technology', 'market', 'apple',
                  'police', 'network', 'service', 'tv', 'media', 'political', 'jan', 'camera', 'music', 'iphone',
                  'online', 'party', '2017', 'afp', 'prime', 'customers', 'korea', 'north', 'users', 'power',
                  'government', 'court', 'prime minister', 'internet', 'game', 'train', 'facebook', 'minister',
                  'advertisement', 'military', 'president', 'business', 'former', 'billion', 'million', 'mobile', 'us',
                  'mr', 'china', 'security']

    ## Tag:	safety_security
    safety_security_pos = ['iraq', 'singpass', 'dead', 'north korea', 'authorities', 'victim', 'victims', 'syrian',
                           'information', 'police said', 'said', 'police', 'personal', 'media', 'agency', 'state',
                           'attack', 'investigation', 'suspect', 'forces', 'islamic state', 'afp', 'injured', 'men',
                           'north', 'death', 'fire', 'bus', 'syria', 'gas', 'missile', 'islamic', 'incident', 'train',
                           'facebook', 'advertisement', 'volcano', 'report', 'alert', 'killed', 'died', 'man',
                           'passengers', 'accident', 'isis', 'attacks', 'island', 'arrested', 'military', 'security']
    safety_security_neg = ['says', 'trump', 'myanmar', 'app', 'global', 'sales', 'trade', 'brain', 'growth', 'per',
                           'go', 'smoking', 'talks', 'market', 'apple', 'even', 'different', 'chinese', 'eu', 'help',
                           'per cent', 'cent', 'research', 'better', 'treatment', 'new', 'meeting', 'economy', 'good',
                           'lee', 'power', 'get', 'food', 'million', 'china', 'president', 'years', 'dr', 'billion',
                           'samsung', 'whether', 'study', 'countries', 'disease', 'deal', 'project', 'patients',
                           'pollution', 'decision', 'first']

    ## Tag:	environment
    environment_pos = ['volcanic', 'disaster', 'burning', 'climate change', 'energy', 'recycling', 'fires', 'weather',
                       'fishing', 'winds', 'deg c', 'species', 'bags', 'air quality', 'paris', 'greenpeace', 'plastic',
                       'environment', 'flights', 'storm', 'waste', 'earthquakes', 'global', 'hit', 'fire', 'eruption',
                       'gas', 'rain', 'water', 'environmental', 'psi', 'santa', 'ash', 'earthquake', 'change', 'homes',
                       'climate', 'indonesia', 'plastic bags', 'floods', 'mount', 'air pollution', 'agung', 'air',
                       'airport', 'emissions', 'island', 'volcano', 'deg', 'pollution']
    environment_neg = ['apple', 'cancer', 'app', 'life', 'years', 'north korea', 'drug', 'family', 'go', 'children',
                       'market', 'united', 'police', 'chinese', 'data', 'support', 'get', 'food', 'attack', 'online',
                       'party', 'korean', 'body', 'myanmar', 'korea', 'north', 'users', 'singapore', 'may', 'court',
                       'missile', 'virus', 'blood', 'men', 'new', 'group', 'cases', 'dr', 'social', 'like', 'mobile',
                       'medical', 'nuclear', 'foreign', 'patients', 'dengue', 'mr', 'military', 'security', 'first']

    ## Tag:	social_relations
    social_relations_pos = ['iran', 'trump', 'myanmar', 'trade', 'states', 'talks', 'britain', 'summit', 'eu', 'japan',
                            'nations', 'religious', 'islam', 'south', 'united', 'chinese', 'jerusalem', 'relations',
                            'neighbours', 'un', 'forces', 'meeting', 'war', 'trumps', 'korea', 'north', 'statement',
                            'government', 'muslim', 'missile', 'islamic', 'defence', 'china', 'minister', 'ethnic',
                            'president', 'united states', 'bangladesh', 'north korea', 'sanctions', 'rakhine', 'rights',
                            'duterte', 'us', 'foreign', 'rohingya', 'saudi', 'military', 'russia', 'muslims']
    social_relations_neg = ['apple', 'cancer', 'app', 'tax', 'sales', 'one', 'four', 'video', 'cent', 'year', 'sexual',
                            'risk', 'use', 'service', 'hospital', 'per cent', 'per', 'according', 'health', 'online',
                            'ebola', 'dengue', 'women', 'cause', 'mugabe', 'available', 'users', 'get', 'food',
                            'camera', 'company', 'water', 'virus', 'facebook', 'blood', 'new', 'services', 'cases',
                            'data', 'zika', 'samsung', 'mobile', 'industry', 'medical', 'disease', 'times', 'patients',
                            'stories', 'found', 'study']

    ## Tag:	meaning_in_life
    meaning_in_life_pos = ['sentenced', 'pakistan', 'moore', 'family', 'allegations', 'elderly', 'accused', 'victim',
                           'claimed', 'misconduct', 'harassment', 'sexual', 'milk', 'rape', 'speak', 'suicide',
                           'police', 'alcohol', 'pet', 'fined', 'overtime', 'bush', 'investigation', 'parents',
                           'luxury', 'prison', 'assault', 'women', 'woman', 'victims', 'allegedly', 'stomp', 'million',
                           'boys', 'saipov', 'abuse', 'child', 'report', 'court', 'former', 'man', 'counselling',
                           'weinstein', 'arrested', 'stomp contributor', 'rohingya', 'pay', 'alabama', 'clothes',
                           'charged']
    meaning_in_life_neg = ['apple', 'photo', 'national', 'head', 'north korea', 'brain', 'experience', 'eu', '5',
                           'risk', 'trump', 'use', 'still', 'nuclear', 'consumers', 'dr', 'drivers', 'get', 'system',
                           'research', 'election', 'health', 'new', 'app', 'exercise', 'virus', 'customers', 'korea',
                           'north', 'users', 'business', 'singapore', 'game', 'facebook', 'plan', 'party', 'data',
                           'zika', 'mobile', 'university', 'disease', 'us', 'airport', 'products', 'south', 'mr',
                           'china', 'study', 'researchers', 'travel']

    ## categories_keywords dict creation
    category_keywords['achievement']['pos'] = achievement_pos;
    category_keywords['achievement']['neg'] = achievement_neg;

    category_keywords['economics']['pos'] = economics_pos;
    category_keywords['economics']['neg'] = economics_neg;

    category_keywords['politics']['pos'] = politics_pos;
    category_keywords['politics']['neg'] = politics_neg;

    category_keywords['na']['pos'] = na_pos;
    category_keywords['na']['neg'] = na_neg;

    category_keywords['skip']['pos'] = skip_pos;
    category_keywords['skip']['neg'] = skip_neg;

    category_keywords['health']['pos'] = health_pos;
    category_keywords['health']['neg'] = health_neg;

    category_keywords['safety_security']['pos'] = safety_security_pos;
    category_keywords['safety_security']['neg'] = safety_security_neg;

    category_keywords['environment']['pos'] = environment_pos;
    category_keywords['environment']['neg'] = environment_neg;

    category_keywords['social_relations']['pos'] = social_relations_pos;
    category_keywords['social_relations']['neg'] = social_relations_neg;

    category_keywords['meaning_in_life']['pos'] = meaning_in_life_pos;
    category_keywords['meaning_in_life']['neg'] = meaning_in_life_neg;

    uniqueKeyWords = set(
        achievement_pos + achievement_neg + economics_pos + economics_neg + politics_pos + politics_neg + na_pos + na_neg + skip_pos + skip_neg + \
        health_pos + health_neg + safety_security_pos + safety_security_neg + environment_pos + environment_neg + social_relations_pos \
        + social_relations_neg + meaning_in_life_pos + meaning_in_life_neg)

    final_data = dict()
    glove_risk_dict = utility.getGloveVec(uniqueKeyWords)
    final_data['category_keywords'] = category_keywords;
    final_data['glove_risk_dict'] = glove_risk_dict

    ## Writing TWEET Data
    with open(risk_cat_file, 'w') as f:
        json.dump(final_data, f)




def getClassMap():
    class_mapping = {
        91: 'health', 92: 'safety_security', 93: 'environment',
        94: 'social_relations', 95: 'meaning_in_life', 96: 'achievement',
        97: 'economics', 98: 'politics', 99: 'not_applicable', 0: 'skip'}
    return class_mapping

if __name__=='__main__':
    main()


