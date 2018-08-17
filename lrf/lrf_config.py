import os

def get_class_map():
    class_mapping = {
        'health': 91, 'safety_security' : 92, 'environment' : 93,
        'social_relations' : 94, 'meaning_in_life' : 95, 'achievement': 96,
        'economics' :97 , 'politics': 98, 'not_applicable': 99, 'skip': 0}
    return class_mapping

def get_char_dict():
    char_map = {'A' : 0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25,
                'a':26,'b':27,'c':28,'d':29,'e':30,'f':31,'g':32,'h':33,'i':34,'j':35,'k':36,'l':37,'m':38,'n':39,'o':40,'p':41,'q':42,'r':43,'s':44,'t':45,'u':46,'v':47,'w':48,'x':49,'y':50,'z':51,
                '0':52,'1':53,'2':54,'3':55,'4':56,'5':57,'6':58,'7':59,'8':60,'9':61,
                '!':62,'\\':63,'"':64,'#':65,'$':66,'%':67,'&':68,"'":69,'(':70,')':71,'*':72,'+':73,'-':74,'.':75,'/':76,':':77,';':78,'<':79,'=':80,'>':81,'?':82,'@':83,'[':84,']':85,'^':86,'_':87,'`':88,'{':89,'|':90,'}':91,'~':92,',':93}
    return char_map

def get_orthohraphic_char_dict():
    ortho_char_map = {'C':0,'c':1,'n':2,'p':3}
    return ortho_char_map

def get_sentiment_map():
    sentiment_map = {
        'negative' : 0, 'neutral' : 1, 'positive' : 2
    }
    return sentiment_map
def get_intensifier_and_diminishers():
    intensifiers = ['absolutely','amazingly','completely','entirely','especially','extremely','freaking','incredibly','fucking','lot','really','super','very']
    diminishers = ['barely','bit','few','hardly','less','nearly','negligibly','partly','practically','rarely','some','sparsely']

    return intensifiers,diminishers

def get_locations():

    locations = {}
    ## Get the Paths
    LRF_PATH = os.getcwd()
    PROJ_PATH = os.path.abspath(os.path.join(LRF_PATH, os.pardir))
    print(LRF_PATH)
    REF_DATA_PATH = os.path.join(PROJ_PATH, 'reference_data/')
    INTERMED_DATA_PATH = os.path.join(PROJ_PATH, 'intermed_data/')

    locations['LRF_PATH'] = LRF_PATH
    locations['PROJ_PATH'] = PROJ_PATH
    locations['REF_DATA_PATH'] = REF_DATA_PATH
    locations['INTERMED_DATA_PATH'] = INTERMED_DATA_PATH

    return locations


def get_file_name():

    ## File Names
    SENTIMENT_FILE_NAME = 'sentiment_lexicon.txt'
    KEYWORD_FILE = 'annotator_data_dump_with_text'
    GLOVE_25D_FILE = 'glove.twitter.27B.25d.txt'
    GLOVE_50D_FILE = 'glove.twitter.27B.50d.txt'
    GLOVE_100D_FILE = 'glove.twitter.27B.100d.txt'
    GLOVE_200D_FILE = 'glove.twitter.27B.200d.txt'

