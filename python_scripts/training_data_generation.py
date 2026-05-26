import copy
import gc
import json
import os
import random

from sim_pre_computation import get_desc_dicts
from utility import read_groundtruth_with_mode, get_triple_dicts


def read_llm_ea_output(dataset_name: str, llm_mode: str, ea_data_mode: str, triple_sel: str,
                       exp_sel: str, tar_sel: str, ptm_model: str, llm_type: str, llm_temp: str,
                       rep: int, lower_bound: int, upper_bound: int, output_mode: str):
    if output_mode == 'single':
        output_dir = os.path.join(os.getcwd(), '..', 'output', 'ea_result', dataset_name, 'processed')
    elif output_mode == 'distribution':
        output_dir = os.path.join(os.getcwd(), '..', 'output', 'ea_result', dataset_name, 'distribution')
    else:
        raise RuntimeError('Unknown output_mode: %s' % output_mode)

    output_file_name = 'train_distribution=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.json' % \
                       (llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel,
                        ptm_model, llm_type, llm_temp, rep, lower_bound, upper_bound)
    output_file_path = os.path.join(output_dir, output_file_name)
    with open(output_file_path, 'r', encoding='utf-8') as f:
        output_dict = json.load(f)

    return output_dict


def select_neg_ent_randomly(pos_ent: str, ent2_list: list):
    neg_ent = random.choice(ent2_list)

    while neg_ent == pos_ent:
        neg_ent = random.choice(ent2_list)

    return neg_ent


def generate_training_data_from_single_output(dataset_name: str, llm_mode: str, ea_data_mode: str, triple_sel: str,
                                              exp_sel: str, tar_sel: str, ptm_model: str, llm_type: str, llm_temp: str,
                                              rep: int, lower_bound: int, upper_bound: int):
    output_mode = 'single'
    ea_output_dict = read_llm_ea_output(dataset_name, llm_mode, ea_data_mode, triple_sel,
                                        exp_sel, tar_sel, ptm_model, llm_type, llm_temp,
                                        rep, lower_bound, upper_bound, output_mode)

    gt_dict = read_groundtruth_with_mode(dataset_name, ea_data_mode)
    ent2_list = list(gt_dict.values())
    desc_model = 'sparql'
    ent1_desc_dict, ent2_desc_dict = get_desc_dicts(dataset_name, 'all', desc_model)

    training_data_dict = {}
    num_error = 0
    for ent1 in ea_output_dict:
        ent1_desc = ent1_desc_dict[ent1]
        ent1_iri_desc_pair = '%s;;;%s' % (ent1, ent1_desc)
        pos_ent = ea_output_dict[ent1]
        if pos_ent not in ent2_desc_dict:
            num_error += 1
            print('[error: %d]llm output not in ent2 list: (%s-%s)' % (num_error, ent1, pos_ent))
            continue
        pos_desc = ent2_desc_dict[pos_ent]
        pos_iri_desc_pair = '%s;;;%s' % (pos_ent, pos_desc)
        neg_ent = select_neg_ent_randomly(pos_ent, ent2_list)
        neg_desc = ent2_desc_dict[neg_ent]
        neg_iri_desc_pair = '%s;;;%s' % (neg_ent, neg_desc)
        training_data_dict[ent1_iri_desc_pair] = (pos_iri_desc_pair, neg_iri_desc_pair)

    training_data_dir = os.path.join(os.getcwd(), '..', 'output', 'training_data', dataset_name)
    if not os.path.exists(training_data_dir):
        os.makedirs(training_data_dir)
    training_data_file_name = 'training_data_single=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.json' % \
                              (llm_mode, ea_data_mode, triple_sel, exp_sel, tar_sel,
                               ptm_model, llm_type, llm_temp, rep, lower_bound, upper_bound)
    training_data_file_path = os.path.join(training_data_dir, training_data_file_name)
    with open(training_data_file_path, 'w', encoding='utf-8') as f:
        json.dump(training_data_dict, f, ensure_ascii=False, indent=4)
    print('Training data generated successfully...%s' % training_data_file_path)

    return 0


def sort_hit_list_fixing_pos_index(hit_list: list, pos_index: int):
    # Separate the specific item from the list
    copyed_hit_list = copy.deepcopy(hit_list)
    pos_value = copyed_hit_list.pop(pos_index)

    # Sort the remaining items in descending order
    sorted_hit_list = sorted(copyed_hit_list, reverse=True)

    # Insert the specific item back into its original position
    sorted_hit_list.insert(pos_index, pos_value)

    return sorted_hit_list


def select_pos_neg_from_distribution(hit_list: list):
    # Find the positive element (max value)
    max_value = max(hit_list)
    pos_index = hit_list.index(max_value)

    sorted_hit_list = sort_hit_list_fixing_pos_index(hit_list, pos_index)
    # Find the negative element
    neg_index = -1
    for i in range(len(hit_list)):
        if i != pos_index:
            cur_value = hit_list[i]
            golden_value = sorted_hit_list[i]
            if cur_value != golden_value:
                neg_index = i
                break

    # If all elements are in correct descending order, select a random one except the positive element
    if neg_index == -1:
        for i in range(len(hit_list)):
            if i != pos_index:
                neg_index = i
                break

    return pos_index, neg_index


def generate_training_data_from_distribution(dataset_name: str, llm_mode: str, ea_data_mode: str, triple_sel: str,
                                             exp_sel: str, tar_sel: str, ptm_model: str, llm_type: str, llm_temp: str,
                                             max_rep: int, lower_bound: int, upper_bound: int, train_percentage: float):
    training_data_dir = os.path.join(os.getcwd(), '..', 'output', 'training_data', dataset_name)
    if not os.path.exists(training_data_dir):
        os.makedirs(training_data_dir)
    train_bound = int(upper_bound * train_percentage)
    training_data_file_name = 'training_data_distribution=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.json' % \
                              (llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel,
                               ptm_model, llm_type, llm_temp, max_rep, lower_bound, train_bound)
    training_data_file_path = os.path.join(training_data_dir, training_data_file_name)

    print("Generating training data...%s" % training_data_file_path)

    output_mode = 'distribution'
    ea_output_train_dict = read_llm_ea_output(dataset_name, llm_mode, ea_data_mode, triple_sel,
                                              exp_sel, tar_sel, ptm_model, llm_type, llm_temp,
                                              max_rep, lower_bound, train_bound, output_mode)
    triple_sel_splits = triple_sel.split('-')
    element_type = triple_sel_splits[2]
    if element_type == 'desc':
        desc_model = 'sparql'
        ent1_elm_dict, ent2_elm_dict = get_desc_dicts(dataset_name, 'all', desc_model)
    elif element_type == 'triple':
        num_triple = int(triple_sel_splits[1])
        triple_strategy = triple_sel_splits[3]
        ent1_elm_dict, ent2_elm_dict = get_triple_dicts(dataset_name, ea_data_mode, num_triple, triple_strategy)
    else:
        raise RuntimeError('Unsupported element type: %s' % element_type)

    training_data_dict = {}  # Initialize an empty list to hold all JSON objects
    num_error = 0
    for ent1 in ea_output_train_dict:
        ent1_elm = ent1_elm_dict[ent1]
        ent1_iri_elm_pair = '%s;;;%s' % (ent1, ent1_elm)
        ent2_hit_pairs = ea_output_train_dict[ent1]
        ent2_hits = []
        for ent2_hit_pair in ent2_hit_pairs:
            ent2_hit = ent2_hit_pair[1]
            ent2_hits.append(ent2_hit)

        pos_index, neg_index = select_pos_neg_from_distribution(ent2_hits)
        pos_ent = ea_output_train_dict[ent1][pos_index][0]
        if pos_ent not in ent2_elm_dict:
            num_error += 1
            print('[error: %d]llm output not in ent2 list: (%s-%s)' % (num_error, ent1, pos_ent))
            continue
        pos_desc = ent2_elm_dict[pos_ent]
        pos_iri_desc_pair = '%s;;;%s' % (pos_ent, pos_desc)
        neg_ent = ea_output_train_dict[ent1][neg_index][0]
        if neg_ent not in ent2_elm_dict:
            num_error += 1
            print('[error: %d]llm output not in ent2 list: (%s-%s)' % (num_error, ent1, neg_ent))
            continue
        neg_desc = ent2_elm_dict[neg_ent]
        neg_iri_desc_pair = '%s;;;%s' % (neg_ent, neg_desc)
        # Create a dictionary for the positive and negative examples
        training_data_entry = {
            ent1_iri_elm_pair: (pos_iri_desc_pair, neg_iri_desc_pair)
        }
        # Append the dictionary to the list
        training_data_dict.update(training_data_entry)
        # Clear variables to reduce memory usage
        del ent1, ent1_elm, ent1_iri_elm_pair, ent2_hit_pairs, ent2_hits, \
            pos_index, pos_ent, pos_desc, pos_iri_desc_pair, \
            neg_index, neg_ent, neg_desc, neg_iri_desc_pair

    with open(training_data_file_path, 'w', encoding='utf-8') as f:
        json.dump(training_data_dict, f, ensure_ascii=False, indent=4)
    print('Training data generated successfully...%s' % training_data_file_path)

    # Clear the large dictionary if it's no longer needed
    del ea_output_train_dict, training_data_dict

    return 0


if __name__ == '__main__':
    ptm_model = 'mpnet'
    # range of test cases
    lower_bound = 0
    upper_bound = 3000
    dataset_name = 'DBP15K_DE_EN_V1'
    llm_mode = 'zs'
    ea_data_mode = 'train-20'
    triple_sel = 'freq-5'
    exp_sel = ''
    tar_sel = 'ptm-5-desc-sparql'
    ## gpt3.5, gpt4, baichuan2-v100, baichuan2-a6000
    llm_type = 'gpt3.5'
    llm_temp = '0'
    # rep = 0
    # generate_training_data_from_single_output(dataset_name, llm_mode, ea_data_mode, num_triple, exp_sel, tar_sel,
    #                                           ptm_model, llm_type, llm_temp, rep, lower_bound, upper_bound)

    max_rep = 5
    train_percentage = 0.9
    generate_training_data_from_distribution(
        dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel, tar_sel, ptm_model, llm_type, llm_temp,
        max_rep, lower_bound, upper_bound, train_percentage)
