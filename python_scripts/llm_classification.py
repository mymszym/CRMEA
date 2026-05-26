import copy
import random
import time

from llm_call import get_response_from_llm
from utility import read_ent_first_rel_triples, read_ent_first_attr_triples, read_groundtruth_with_mode, \
    get_element_triples, get_triple_dicts, get_attr_and_rel_filenames


def get_ent_tuples(ent: str, triples: dict):
    tuples = []
    if ent in triples:
        tuples = triples[ent]

    return tuples


def get_random_samples_from_dict(input_dict: dict, n: int):
    # Randomly select n keys from the dictionary
    selected_keys = random.sample(list(input_dict.keys()), n)
    # Create a new dictionary using the selected keys
    random_dict = {key: input_dict[key] for key in selected_keys}

    return random_dict


def add_ent_triple_to_prompt(prompt: str, dataset_name: str, ea_data_mode: str, triple_sel: str,
                             tar_ent: str, tar_ent_list: str):
    triple_sel_splits = triple_sel.split('-')
    triple_strategy = triple_sel_splits[3]
    num_triple = int(triple_sel_splits[1])

    prompt += '[triples]: ['

    ent_triple_text = ''
    if triple_strategy == 'freq':
        ent1_triple_dict, ent2_triple_dict = get_triple_dicts(dataset_name, ea_data_mode, num_triple, triple_strategy)
        if tar_ent_list == 'ent1':
            ent_triple_dict = ent1_triple_dict
        elif tar_ent_list == 'ent2':
            ent_triple_dict = ent2_triple_dict
        else:
            raise RuntimeError('Unknown target_ent_list: %s' % tar_ent_list)

        if tar_ent in ent_triple_dict.keys():
            ent_triple_list = ent_triple_dict[tar_ent]
            for ent_triple in ent_triple_list:
                ent_triple_text += ent_triple + ', '
    elif triple_strategy == 'rand':
        ent_name = tar_ent.split("/")[-1]
        attr_filename, rel_filename = get_attr_and_rel_filenames(dataset_name, tar_ent_list)
        attr_dict = read_ent_first_attr_triples(dataset_name, attr_filename)
        if tar_ent in attr_dict.keys():
            ent_attr_dict = attr_dict[tar_ent]
            if num_triple < len(ent_attr_dict):
                ent_attr_dict = get_random_samples_from_dict(ent_attr_dict, num_triple)
            for attr in ent_attr_dict:
                attr_name = attr.split("/")[-1]
                value = ent_attr_dict[attr][0]
                ent_triple_text += '(%s, %s, %s), ' % (ent_name, attr_name, value)
        rel_out_dict, rel_in_dict = read_ent_first_rel_triples(dataset_name, rel_filename)
        if tar_ent in rel_out_dict.keys():
            ent_rel_out_dict = rel_out_dict[tar_ent]
            if num_triple < len(ent_rel_out_dict):
                ent_rel_out_dict = get_random_samples_from_dict(ent_rel_out_dict, num_triple)
            for rel_out in ent_rel_out_dict:
                rel_out_name = rel_out.split("/")[-1]
                object = ent_rel_out_dict[rel_out][0]
                object_name = object.split("/")[-1]
                ent_triple_text += '(%s, %s, %s), ' % (ent_name, rel_out_name, object_name)
        if tar_ent in rel_in_dict.keys():
            ent_rel_in_dict = rel_in_dict[tar_ent]
            if num_triple < len(ent_rel_in_dict):
                ent_rel_in_dict = get_random_samples_from_dict(ent_rel_in_dict, num_triple)
            for rel_in in ent_rel_in_dict:
                rel_in_name = rel_in.split("/")[-1]
                suject = ent_rel_in_dict[rel_in][0]
                suject_name = suject.split("/")[-1]
                ent_triple_text += '(%s, %s, %s), ' % (suject_name, rel_in_name, ent_name)
    else:
        raise RuntimeError('Unknown triple_strategy: %s' % triple_strategy)

    if len(ent_triple_text) > 2:
        ent_triple_text = ent_triple_text[:-2]
    prompt += ent_triple_text
    prompt += ']\n'

    return prompt


def get_domain_from_iri(ent: str):
    slash_splits = ent.split("/")
    # from "http://dbpedia.org/resource/Rock_music" extract domain "http://dbpedia.org"
    domain = slash_splits[0] + '//' + slash_splits[2]

    return domain


# def generate_icl_exp(dataset_name: str, ea_data_mode: str, exp_sel: str, ent1: str,
#                      ent1_triple_dict: dict, ent2_triple_dict: dict):
#     icl_exp = 'Here are some EA examples:\n'
#     gt_dict = read_groundtruth_with_mode(dataset_name, ea_data_mode)
#     exp_splits = exp_sel.split('-')
#     exp_strategy = exp_splits[0]
#     num_exp = int(exp_splits[1])
#     if exp_strategy == 'random':
#         temp_dict = copy.deepcopy(gt_dict)
#         del temp_dict[ent1]
#         items = list(temp_dict.items())
#         example_pairs = random.choices(items, k=num_exp)
#         index = 0
#         for example in example_pairs:
#             index += 1
#             sor_ent = example[0]
#             icl_exp += '[[source entity%d]] : [[%s]]\n' % (index, sor_ent)
#             icl_exp = add_ent_triple_to_prompt(icl_exp, sor_ent, ent1_triple_dict)
#             align_ent = example[1]
#             icl_exp += '[[aligned entity%d]] : [[%s]]\n' % (index, align_ent)
#             icl_exp = add_ent_triple_to_prompt(icl_exp, align_ent, ent2_triple_dict)
#     else:
#         raise RuntimeError('Unknown example selection strategy: %s' % exp_strategy)
#
#     return icl_exp


def prompt_generation(dataset_name: str, llm_mode: str, ea_data_mode: str, triple_sel: str, exp_sel: str,
                      ent1: str, ent2_list: list):
    instruction = '**Instruction:** Given a source entity and several candidate target entities, ' \
                  'find the target entity that matches the same real-world object as the source. ' \
                  'Each entity is described by triples like <subject, predicate, object> (e.g., ' \
                  '<Paris, isCapitalOf, France>). Use both the entity name and triples to decide. ' \
                  'Answer with the exact IRI of the best match, nothing else.\n'
    if llm_mode == 'zs':
        icl_exp = ''
    elif llm_mode == 'icl':
        # icl_exp = generate_icl_exp(dataset_name, ea_data_mode, exp_sel, ent1, ent1_triple_dict, ent2_triple_dict)
        if ent1 != 'http://dbpedia.org/resource/Marco_Polo_(miniseries)':
            icl_exp = '**Demonstration:** Here is an example\n' \
                      '**Source Entity**\n' \
                      '[IRI]: [http://dbpedia.org/resource/Marco_Polo_(miniseries)]\n' \
                      '[triples]: [(Marco_Polo_(miniseries), name, Marco Polo), ' \
                      '(Marco_Polo_(miniseries), runtime, ""32700.0""^^<http://www.w3.org/2001/XMLSchema#double>), ' \
                      '(Marco_Polo_(miniseries), numberOfEpisodes, ""8""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), ' \
                      '(Marco_Polo_(miniseries), numberOfSeasons, ""1""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), ' \
                      '(Marco_Polo_(miniseries), starring, Denholm_Elliott), ' \
                      '(Marco_Polo_(miniseries), location, Morocco), ' \
                      '(Marco_Polo_(miniseries), composer, Ennio_Morricone)]\n' \
                      '**Target Entity 1**\n' \
                      '[IRI]: [http://de.dbpedia.org/resource/Marco_Polo_(1982)]\n' \
                      '[triples]: [(Marco_Polo_(1982), name, Marco Polo), ' \
                      '(Marco_Polo_(1982), musicComposer, Ennio_Morricone), ' \
                      '(Marco_Polo_(1982), starring, Anne_Bancroft)]\n' \
                      '**Target Entity 2**\n' \
                      '[IRI]: [http://de.dbpedia.org/resource/Marco_Haber]\n' \
                      '[triples]: [(Marco_Haber, appearancesInLeague, ""9""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), ' \
                      '(Marco_Haber, years, ""1989""^^<http://www.w3.org/2001/XMLSchema#gYear>), ' \
                      '(Marco_Haber, height, ""1.82""^^<http://www.w3.org/2001/XMLSchema#double>), ' \
                      '(Marco_Haber, birthDate, ""1971-09-21""^^<http://www.w3.org/2001/XMLSchema#date>), ' \
                      '(Marco_Haber, youthYears, ""1985""^^<http://www.w3.org/2001/XMLSchema#gYear>), ' \
                      '(Marco_Haber, club, SpVgg_Unterhaching)]\n' \
                      '**Target Entity 3**\n' \
                      '[IRI]: [http://de.dbpedia.org/resource/Marco_Pezzaiuoli]\n' \
                      '[triples]: [(Marco_Pezzaiuoli, trainerYears, ""2000""^^<http://www.w3.org/2001/XMLSchema#gYear>), ' \
                      '(Marco_Pezzaiuoli, name, Pezzaiuoli, Marco), ' \
                      '(Marco_Pezzaiuoli, birthDate, ""1968-11-16""^^<http://www.w3.org/2001/XMLSchema#date>), ' \
                      '(Marco_Pezzaiuoli, trainerClub, TSG_1899_Hoffenheim), ' \
                      '(Marco_Pezzaiuoli, managerClub, TSG_1899_Hoffenheim), (Marco_Pezzaiuoli, club, VfR_Mannheim), ' \
                      '(Marco_Pezzaiuoli, birthPlace, Mannheim)]\n' \
                      '**Target Entity 4**\n' \
                      '[IRI]: [http://de.dbpedia.org/resource/Marco_Grimm]\n' \
                      '[triples]: [(Marco_Grimm, height, ""1.87""^^<http://www.w3.org/2001/XMLSchema#double>), ' \
                      '(Marco_Grimm, name, Marco Grimm), ' \
                      '(Marco_Grimm, appearancesInLeague, ""26""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), ' \
                      '(Marco_Grimm, birthDate, ""1972-06-16""^^<http://www.w3.org/2001/XMLSchema#date>), ' \
                      '(Marco_Grimm, years, ""2003""^^<http://www.w3.org/2001/XMLSchema#gYear>), ' \
                      '(Marco_Grimm, club, Eintracht_Braunschweig), (Marco_Grimm, birthPlace, Baden-Württemberg)]\n'\
                      '**Target Entity 5**\n' \
                      '[IRI]: [http://de.dbpedia.org/resource/Marco_Beltrami]\n' \
                      '[triples]: [(Marco_Beltrami, name, Beltrami, Marco), ' \
                      '(Marco_Beltrami, nick, Beltrami, Marco Edward (vollständiger Name)), ' \
                      '(Marco_Beltrami, birthDate, ""1966-10-07""^^<http://www.w3.org/2001/XMLSchema#date>), ' \
                      '(Marco_Beltrami, viafId, 100229084), (Marco_Beltrami, lccn, n/97/854793), ' \
                      '(Marco_Beltrami, birthPlace, Long_Island), (Angel_Eyes, musicComposer, Marco_Beltrami)]\n' \
                      '**Answer:** [http://de.dbpedia.org/resource/Marco_Polo_(1982)]\n'
        else:
            icl_exp = '**Demonstration:** Here is an example:\n' \
                      '**Source Entity**\n' \
                      '[IRI]: [http://dbpedia.org/resource/Carlos_Bianchi]\n' \
                      '[triples]: [(Carlos_Bianchi, name, Carlos Bianchi), ' \
                      '(Carlos_Bianchi, birthDate, ""1949-04-26""^^<http://www.w3.org/2001/XMLSchema#date>), ' \
                      '(Carlos_Bianchi, height, ""0.0""^^<http://www.w3.org/2001/XMLSchema#double>), ' \
                      '(Carlos_Bianchi, managerClub, A.S._Roma), (Carlos_Bianchi, team, Club_Atlético_Vélez_Sarsfield), ' \
                      '(Carlos_Bianchi, birthPlace, Buenos_Aires)]\n' \
                      '**Target Entity 1**\n' \
                      '[IRI]: [http://de.dbpedia.org/resource/Carlos_Bianchi]\n' \
                      '[triples]: [(Carlos_Bianchi, youthYears, ""1973""^^<http://www.w3.org/2001/XMLSchema#gYear>), ' \
                      '(Carlos_Bianchi, birthDate, ""1949-04-26""^^<http://www.w3.org/2001/XMLSchema#date>), ' \
                      '(Carlos_Bianchi, appearancesInNationalTeam, ""14""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), ' \
                      '(Carlos_Bianchi, appearancesInLeague, ""74""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), ' \
                      '(Carlos_Bianchi, years, ""1973""^^<http://www.w3.org/2001/XMLSchema#gYear>), ' \
                      '(Carlos_Bianchi, trainerClub, AS_Rom), (Carlos_Bianchi, managerClub, CA_Vélez_Sársfield), ' \
                      '(Carlos_Bianchi, club, Stade_Reims), (Carlos_Bianchi, birthPlace, Buenos_Aires)]\n' \
                      '**Target Entity 2**\n' \
                      '[IRI]: [http://de.dbpedia.org/resource/Javier_Irureta]\n' \
                      '[triples]: [(Javier_Irureta, trainerYears, ""2008""^^<http://www.w3.org/2001/XMLSchema#gYear>), ' \
                      '(Javier_Irureta, appearancesInLeague, ""208""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), ' \
                      '(Javier_Irureta, nick, Iruretagoyena Amianó, Javier), ' \
                      '(Javier_Irureta, nationalYears, ""1979""^^<http://www.w3.org/2001/XMLSchema#gYear>), ' \
                      '(Javier_Irureta, appearancesInNationalTeam, ""1""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), ' \
                      '(Javier_Irureta, trainerClub, Real_Saragossa), (Javier_Irureta, managerClub, Real_Saragossa), ' \
                      '(Javier_Irureta, club, Athletic_Bilbao)]\n' \
                      '**Target Entity 3**\n' \
                      '[IRI]: [http://de.dbpedia.org/resource/Carlos_Bilardo]\n' \
                      '[triples]: [(Carlos_Bilardo, birthDate, ""1939-03-16""^^<http://www.w3.org/2001/XMLSchema#date>), ' \
                      '(Carlos_Bilardo, nick, Bilardo, Carlos Salvador), (Carlos_Bilardo, name, Carlos Salvador Bilardo), ' \
                      '(Carlos_Bilardo, trainerYears, ""1997""^^<http://www.w3.org/2001/XMLSchema#gYear>), ' \
                      '(Carlos_Bilardo, nationalYears, ""1960""^^<http://www.w3.org/2001/XMLSchema#gYear>), ' \
                      '(Carlos_Bilardo, birthPlace, Buenos_Aires), (Carlos_Bilardo, club, CA_San_Lorenzo_de_Almagro), ' \
                      '(Carlos_Bilardo, trainerClub, Kolumbianische_Fußballnationalmannschaft), ' \
                      '(Carlos_Bilardo, managerClub, Estudiantes_de_La_Plata)]\n' \
                      '**Target Entity 4**\n' \
                      '[IRI]: [http://de.dbpedia.org/resource/Vicente_Miera]\n' \
                      '[triples]: [(Vicente_Miera, name, Vicente Miera Campos), ' \
                      '(Vicente_Miera, trainerYears, ""1986""^^<http://www.w3.org/2001/XMLSchema#gYear>), ' \
                      '(Vicente_Miera, appearancesInLeague, ""14""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), ' \
                      '(Vicente_Miera, years, ""1960""^^<http://www.w3.org/2001/XMLSchema#gYear>), ' \
                      '(Vicente_Miera, nick, Miera Campos, Vicente), (Vicente_Miera, trainerClub, Real_Oviedo), ' \
                      '(Vicente_Miera, managerClub, CD_Teneriffa), (Vicente_Miera, club, Sporting_Gijón)]\n' \
                      '**Target Entity 5**\n' \
                      '[IRI]: [http://de.dbpedia.org/resource/Juan_Carlos_Lorenzo]\n' \
                      '[triples]: [(Juan_Carlos_Lorenzo, depictionDescription, Juan Carlos Lorenzo (1987)), ' \
                      '(Juan_Carlos_Lorenzo, deathDate, ""2001-11-14""^^<http://www.w3.org/2001/XMLSchema#date>), ' \
                      '(Juan_Carlos_Lorenzo, trainerYears, ""1961""^^<http://www.w3.org/2001/XMLSchema#gYear>), ' \
                      '(Juan_Carlos_Lorenzo, birthDate, ""1922-10-22""^^<http://www.w3.org/2001/XMLSchema#date>), ' \
                      '(Juan_Carlos_Lorenzo, appearancesInLeague, ""79""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), ' \
                      '(Juan_Carlos_Lorenzo, club, Rayo_Vallecano), (Juan_Carlos_Lorenzo, deathPlace, Buenos_Aires), ' \
                      '(Juan_Carlos_Lorenzo, trainerClub, Racing_Club_(Avellaneda)), ' \
                      '(Juan_Carlos_Lorenzo, managerClub, Lazio_Rom), (Juan_Carlos_Lorenzo, birthPlace, Buenos_Aires)]\n' \
                      '[Answer]: [http://de.dbpedia.org/resource/Carlos_Bianchi]\n'
    else:
        raise RuntimeError('Unknown llm mode: %s' % llm_mode)

    sou_ent_iri = '**Query:** The following is the query\n **Source Entity**\n[IRI]: [%s]\n' % ent1
    # kg1_domain = get_domain_from_iri(ent1)
    # sou_ent_domain = '[domain]: [%s]\n' % kg1_domain
    # prompt = instruction + icl_exp + sou_ent_iri + sou_ent_domain
    prompt = instruction + icl_exp + sou_ent_iri
    prompt = add_ent_triple_to_prompt(prompt, dataset_name, ea_data_mode, triple_sel, ent1, 'ent1')

    # kg2_domain = get_domain_from_iri(ent2_list[0])
    # tar_domain = '[target entity domain]: [%s]\n' % kg2_domain
    # prompt += tar_domain
    index = 0
    for ent2 in ent2_list:
        index += 1
        tar_ent_iri = '**Target Entity %d**\n [IRI]: [%s]\n' % (index, ent2)
        prompt += tar_ent_iri
        prompt = add_ent_triple_to_prompt(prompt, dataset_name, ea_data_mode, triple_sel, ent2, 'ent2')

    image_prompt = '''
        Image information is used to enhance entity representation.
        Image 1 is a Source Entity,
        Image 2 corresponds to Target Entity 1,
        Image 3 corresponds to Target Entity 2,
        Image 4 corresponds to Target Entity 3,
        Image 5 corresponds to Target Entity 4,
        Image 6 corresponds to Target Entity 5,
        Image 7 corresponds to Target Entity 6,
        Image 8 corresponds to Target Entity 7,
        Image 9 corresponds to Target Entity 8,
        Image 10 corresponds to Target Entity 9,
        Image 11 corresponds to Target Entity 10,
        If the image corresponding to an entity is a pure white image, it means that the entity has no image information, 
        and the blank image is just occupying space. Please use text to represent the information of the entity.


        '''

    prompt += image_prompt


    question = 'Which target entity do you think is the most similar one regarding to the source entity? ' \
               'Please answer with exactly one target entity IRI without any description or square brackets.'
    prompt += question

    return prompt


def llm_classify(dataset_name: str, llm_mode: str, ea_data_mode: str, triple_sel: str, exp_sel: str,
                 ent1: str, ent2_list: list, llm_type: str, llm_model: str, llm_temp: str):
    prompt = prompt_generation(dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel, ent1, ent2_list)

    # start_time = time.time()
    # print(f"ent2_list:{ent2_list}")
    response = get_response_from_llm(llm_type, prompt, llm_model, llm_temp, ent1, ent2_list)
    # response = get_response_from_llm(llm_type, prompt, llm_model, llm_temp)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"execution time：{execution_time} seconds")
    # print(response)
    return response


if __name__ == '__main__':
    dataset_name = 'DBP15K_DE_EN_V1'
    ent1 = 'http://dbpedia.org/resource/Rock_music'
    ent2_list = ['http://de.dbpedia.org/resource/Paulo_de_Almeida_Ribeiro',
                 'http://de.dbpedia.org/resource/Abel_Braga',
                 'http://de.dbpedia.org/resource/Vanderlei_Luxemburgo']
    llm_mode = 'icl'
    ea_data_mode = 'train-20'
    triple_sel = 'freq-5'
    exp_sel = 'random-10'
    prompt = prompt_generation(dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel, ent1, ent2_list)
    print(prompt)

    ## gpt3.5, gpt4, baichuan2-v100
    llm_type = 'gpt3.5'
    llm_model = 'gpt-3.5-turbo-1106'
    llm_temp = '0'
    response = get_response_from_llm(llm_type, prompt, llm_model, llm_temp)
    print(response)
