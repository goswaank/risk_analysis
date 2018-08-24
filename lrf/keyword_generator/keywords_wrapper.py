import os
import json
from lrf.keyword_generator import keyword_generation_micromort as key_gen
from lrf.utilities import utility, risk_categories as rc


def main():
    lrf_path = os.getcwd()
    proj_path = os.path.abspath(os.path.join(lrf_path, os.pardir))
    intermed_data_path = os.path.join(proj_path, 'intermed_data')
    risk_cat_file = os.path.join(intermed_data_path, 'risk_category_file.json')
    glove_key_file = os.path.join(intermed_data_path, 'glove_key_subset.json')
    num_of_keywords = 50
    feature_selection = 'svc'

    class_mapping = rc.getClassMap()

    category_keywords = key_gen.getNewsKeywords(feature_selection,num_of_keywords,class_mapping)

    print(category_keywords)
    unique_keywords = []
    ## To Calculate Unique Keywords:
    for categ in category_keywords:
        unique_keywords = unique_keywords+list(category_keywords[categ]['pos'].keys())
        unique_keywords = unique_keywords+list(category_keywords[categ]['neg'].keys())

    unique_keywords = set(unique_keywords)

    glove_risk_dict = utility.getGloveVec(unique_keywords)

    # final_data = dict()
    # final_data['glove_risk_dict'] = glove_risk_dict
    # final_data['category_keywords'] = category_keywords

    with open(risk_cat_file, 'w') as f:
        json.dump(category_keywords, f)
    with open(glove_key_file, 'w') as f:
        json.dump(glove_risk_dict, f)


if __name__=='__main__':
    main()