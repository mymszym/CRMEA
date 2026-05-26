import json
import os
import shutil

from tqdm import tqdm

from sim_pre_computation import generate_ptm_sim_dict_batch
import re

def get_ptm_model_name(ptm_model: str, round: int):
    if round == 0:
        ptm_model_name = ptm_model
    else:
        ptm_model_name = '%s-round-%d' % (ptm_model, round)

    return ptm_model_name


def read_pred_first_triples(dataset_name: str, file_name: str):
    result_dict = {}
    file_path = os.path.join(os.getcwd(), '..', 'dataset', dataset_name, file_name)
    if not os.path.exists(file_path):
        raise RuntimeError('Input file "%s" does not exist!' % file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if '_att_' in file_name:
                line = line.replace('> <', '\t')
                line = line.replace('> "', '\t')
                line = line.replace('>', '')
                line = line.replace('<', '')

            subj, pred, obj = line.strip().split('\t')
            if pred not in result_dict:
                result_dict[pred] = {subj: [obj]}
            else:
                if subj not in result_dict[pred]:
                    result_dict[pred][subj] = [obj]
                else:
                    result_dict[pred][subj].append(obj)
    return result_dict


def count_elm_first_triples(triples: dict):
    count = 0
    for elm in triples.keys():
        count += len(triples[elm])

    return count


def get_element_triples(dataset_name: str):
    if dataset_name in ['DBP15K_DE_EN_V1', 'DBP15K_FR_EN_V1', 'DBP100K_DE_EN_V1', 'DBP100K_FR_EN_V1',
                        'DW15K_V1', 'DY15K_V1']:
        ent1_attr_file_name = 'attr_triples_1'
        ent2_attr_file_name = 'attr_triples_2'
        ent1_rel_file_name = 'rel_triples_1'
        ent2_rel_file_name = 'rel_triples_2'

    elif dataset_name=='EN_DE_15K_V1':
        ent1_attr_file_name = 'attr_triples_1'
        ent2_attr_file_name = 'attr_triples_2'
        ent1_rel_file_name = 'rel_triples_1'
        ent2_rel_file_name = 'rel_triples_2'
    
    elif dataset_name=='EN_FR_15K_V1':
        ent1_attr_file_name = 'attr_triples_1'
        ent2_attr_file_name = 'attr_triples_2'
        ent1_rel_file_name = 'rel_triples_1'
        ent2_rel_file_name = 'rel_triples_2'

    elif dataset_name == 'DBP15K_FR_EN':
        ent1_attr_file_name = 'fr_att_triples'
        ent2_attr_file_name = 'en_att_triples'
        ent1_rel_file_name = 'fr_rel_triples'
        ent2_rel_file_name = 'en_rel_triples'
    elif dataset_name == 'DBP15K_JA_EN':
        ent1_attr_file_name = 'ja_att_triples'
        ent2_attr_file_name = 'en_att_triples'
        ent1_rel_file_name = 'ja_rel_triples'
        ent2_rel_file_name = 'en_rel_triples'
    elif dataset_name == 'DBP15K_ZH_EN':
        ent1_attr_file_name = 'zh_att_triples'
        ent2_attr_file_name = 'en_att_triples'
        ent1_rel_file_name = 'zh_rel_triples'
        ent2_rel_file_name = 'en_rel_triples'

    elif dataset_name == 'FBDB15K':
        ent1_attr_file_name = './norm/training_attrs_1'
        ent2_attr_file_name = './norm/training_attrs_2'
        ent1_rel_file_name = './norm/triples_1'
        ent2_rel_file_name = './norm/triples_2'
    elif dataset_name == 'fbdb15K':
        ent1_attr_file_name = 'FB15K_NumericalTriples_att_.txt'
        ent2_attr_file_name = 'DB15K_NumericalTriples_att_.txt'
        ent1_rel_file_name = 'FB15K_EntityTriples.txt'
        ent2_rel_file_name = 'DB15K_EntityTriples.txt'


    else:
        raise RuntimeError('Unknown dataset_name: %s' % dataset_name)

    attr_triples1 = read_ent_first_attr_triples(dataset_name, ent1_attr_file_name)
    # print('attr_triples_1: ', attr_triples1)
    rel_out_triples1, rel_in_triples1 = read_ent_first_rel_triples(dataset_name, ent1_rel_file_name)
    # print(11111111111111)
    attr_triples2 = read_ent_first_attr_triples(dataset_name, ent2_attr_file_name)
    # print(attr_triples2)
    rel_out_triples2, rel_in_triples2 = read_ent_first_rel_triples(dataset_name, ent2_rel_file_name)
    # print('rel_out_triples2:', rel_out_triples2)
    return attr_triples1, rel_out_triples1, rel_in_triples1, attr_triples2, rel_out_triples2, rel_in_triples2


def count_element_dict(ent_list: list, element_triples: dict):
    element_count_dict = {}

    for ent in tqdm(ent_list):
        if ent in element_triples:
            ent_element_dict = element_triples[ent]
            for pred in ent_element_dict:
                if pred != 'http://xmlns.com/foaf/0.1/name':
                    if pred not in element_count_dict:
                        element_count_dict[pred] = 1
                    else:
                        element_count_dict[pred] += 1

    sorted_element_list = sorted(element_count_dict, key=lambda k: element_count_dict[k], reverse=True)

    return sorted_element_list


def get_frequent_rel_and_attr(dataset_name: str, ea_data_mode: str):
    gt_dict = read_groundtruth_with_mode(dataset_name, ea_data_mode)

    attr_triples1, rel_out_triples1, rel_in_triples1, attr_triples2, rel_out_triples2, rel_in_triples2 = \
        get_element_triples(dataset_name)

    ent1_list = list(gt_dict.keys())
    ent1_sorted_attr_list = count_element_dict(ent1_list, attr_triples1)
    ent1_sorted_rel_out_list = count_element_dict(ent1_list, rel_out_triples1)
    ent1_sorted_rel_in_list = count_element_dict(ent1_list, rel_in_triples1)

    ent2_list = list(gt_dict.values())
    ent2_sorted_attr_list = count_element_dict(ent2_list, attr_triples2)
    ent2_sorted_rel_out_list = count_element_dict(ent2_list, rel_out_triples2)
    ent2_sorted_rel_in_list = count_element_dict(ent2_list, rel_in_triples2)
    return ent1_sorted_attr_list, ent1_sorted_rel_out_list, ent1_sorted_rel_in_list, \
        ent2_sorted_attr_list, ent2_sorted_rel_out_list, ent2_sorted_rel_in_list


def read_ent_first_rel_triples(dataset_name: str, file_name: str):
    rel_out_dict = {}
    rel_in_dict = {}
    file_path = os.path.join(os.getcwd(), '..', 'dataset', dataset_name, file_name)
    if not os.path.exists(file_path):
        raise RuntimeError('Input file "%s" does not exist!' % file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.endswith('.'):
                line = line[:-1]  # 去掉末尾的点
            parts = line.split()
            if len(parts) >= 3:
                subj = parts[0]
                pred = parts[1]
                obj = ' '.join(parts[2:])  # 处理对象可能包含空格的情况
                if subj not in rel_out_dict:
                    rel_out_dict[subj] = {pred: [obj]}
                else:
                    if pred not in rel_out_dict[subj]:
                        rel_out_dict[subj][pred] = [obj]
                    else:
                        rel_out_dict[subj][pred].append(obj)
                if obj not in rel_in_dict:
                    rel_in_dict[obj] = {pred: [subj]}
                else:
                    if pred not in rel_in_dict[obj]:
                        rel_in_dict[obj][pred] = [subj]
                    else:
                        rel_in_dict[obj][pred].append(subj)

    return rel_out_dict, rel_in_dict


def read_ent_first_rel_triples(dataset_name: str, file_name: str):
    rel_out_dict = {}
    rel_in_dict = {}
    file_path = os.path.join(os.getcwd(), '..', 'dataset', dataset_name, file_name)
    if not os.path.exists(file_path):
        raise RuntimeError('Input file "%s" does not exist!' % file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # DBpedia format ends with ' .'
            if line.endswith(' .'):
                line = line[:-2].strip()

            # Freebase is tab-separated, DBpedia is space-separated
            if '\t' in line:
                parts = line.split('\t', 2)
            else:
                parts = line.split(' ', 2)

            if len(parts) >= 3:
                subj, pred, obj = parts[0], parts[1], parts[2]

                # Clean URIs by removing <> for DBpedia
                if subj.startswith('<') and subj.endswith('>'):
                    subj = subj[1:-1]
                if pred.startswith('<') and pred.endswith('>'):
                    pred = pred[1:-1]
                if obj.startswith('<') and obj.endswith('>'):
                    obj = obj[1:-1]

                if subj not in rel_out_dict:
                    rel_out_dict[subj] = {pred: [obj]}
                else:
                    if pred not in rel_out_dict[subj]:
                        rel_out_dict[subj][pred] = [obj]
                    else:
                        rel_out_dict[subj][pred].append(obj)
                if obj not in rel_in_dict:
                    rel_in_dict[obj] = {pred: [subj]}
                else:
                    if pred not in rel_in_dict[obj]:
                        rel_in_dict[obj][pred] = [subj]
                    else:
                        rel_in_dict[obj][pred].append(subj)

    return rel_out_dict, rel_in_dict


def read_ent_first_attr_triples(dataset_name: str, file_name: str):
    attr_dict = {}
    file_path = os.path.join(os.getcwd(), '..', 'dataset', dataset_name, file_name)
    if not os.path.exists(file_path):
        raise RuntimeError('Input file "%s" does not exist!' % file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            subj, pred, obj = None, None, None

            # Handle DBpedia format: <subj> <pred> "value"^^<type> .
            if line.startswith('<'):
                if line.endswith(' .'):
                    line = line[:-2].strip()

                parts = line.split(' ', 2)
                if len(parts) == 3:
                    subj_part, pred_part, obj_part = parts

                    # Clean subj and pred URI
                    subj = subj_part[1:-1] if subj_part.startswith('<') and subj_part.endswith('>') else subj_part
                    pred = pred_part[1:-1] if pred_part.startswith('<') and pred_part.endswith('>') else pred_part

                    # Extract literal value from "value"^^<type>
                    match = re.search(r'"(.*?)"', obj_part)
                    if match:
                        obj = match.group(1)
                    else:
                        obj = obj_part  # Fallback

            # Handle Freebase/other tab-separated format
            else:
                parts = line.split('\t', 2)
                if len(parts) >= 3:
                    subj, pred, obj = parts[0], parts[1], parts[2]

            if subj and pred and obj is not None:
                if subj not in attr_dict:
                    attr_dict[subj] = {pred: [obj]}
                else:
                    if pred not in attr_dict[subj]:
                        attr_dict[subj][pred] = [obj]
                    else:
                        attr_dict[subj][pred].append(obj)

    return attr_dict

def read_groundtruth_with_mode(dataset_name: str, mode: str):
    if dataset_name in ['DBP15K_DE_EN_V1', 'DBP15K_FR_EN_V1', 'DBP100K_DE_EN_V1', 'DBP100K_FR_EN_V1',
                        'DW15K_V1', 'DY15K_V1']:
        file_name = 'ent_links'

    elif dataset_name in ['EN_DE_15K_V1','EN_FR_15K_V1']:
        file_name = 'ent_links'

    elif dataset_name in ['DBP15K_FR_EN', 'DBP15K_JA_EN', 'DBP15K_ZH_EN']:
        file_name = 'ent_ILLs'

    elif dataset_name in ['FBDB15K']:
        file_name = './norm/ill_ent_ids'
    elif dataset_name in ['fbdb15K']:
        file_name = 'fbdb_link'
    else:
        raise RuntimeError('Unknown dataset name: %s' % dataset_name)

    file_path = os.path.join(os.getcwd(), '..', 'dataset', dataset_name, file_name)
    if not os.path.exists(file_path):
        raise RuntimeError('Input file "%s" does not exist!' % file_path)

    with open(file_path, 'r', encoding='utf-8') as file:
        if mode == 'train-20':
            mode_lb = 0
            if '15K' in dataset_name:
                mode_ub = 3000
            elif '100K' in dataset_name:
                mode_ub = 20000
            else:
                raise RuntimeError('mode_ub calculation error with mode %s for dataset %s' % (mode, dataset_name))
        elif mode == 'test-80':
            if '15K' in dataset_name:
                mode_lb = 3000
                mode_ub = 15000
            elif '100K' in dataset_name:
                mode_lb = 20000
                mode_ub = 100000
            else:
                raise RuntimeError(
                    'mode_ub calculation error with mode %s for dataset %s' % (mode, dataset_name))
        elif mode == 'all':
            mode_lb = 0
            if '15K' in dataset_name:
                mode_ub = 15000
            elif '100K' in dataset_name:
                mode_ub = 100000
            else:
                raise RuntimeError('mode_ub calculation error with mode %s for dataset %s' % (mode, dataset_name))
        else:
            raise RuntimeError('Unknown mode: %s' % mode)

        index = 0
        gt_dict = {}
        for line in file:
            if index < mode_lb:
                index += 1
                continue
            elif index >= mode_ub:
                break
            else:
                ent1, ent2 = line.strip().split('\t')

                # ==================== 新增修改 ====================
                # 检查并移除 ent2 (DBpedia 实体) 的尖括号
                if ent2.startswith('<') and ent2.endswith('>'):
                    ent2 = ent2[1:-1]
                # ent1 是 Freebase ID, 通常不带尖括号, 无需处理
                # =================================================

                if ent1 not in gt_dict:
                    gt_dict[ent1] = ent2
                else:
                    raise RuntimeError('Multiple aligned entites for %s' % ent1)
                index += 1

    return gt_dict


def compute_dataset_statistics(dataset_name: str):
    attr1_filename, rel1_filename = get_attr_and_rel_filenames(dataset_name, 'ent1')
    kg1_ent_out_dict, kg1_ent_in_dict = read_ent_first_rel_triples(dataset_name, rel1_filename)
    ent1_set = set(kg1_ent_out_dict.keys())
    ent1_in_set = set(kg1_ent_in_dict.keys())
    ent1_set.update(ent1_in_set)
    num_ent1 = len(ent1_set)
    kg1_rel_dict = read_pred_first_triples(dataset_name, rel1_filename)
    num_rel1 = len(set(kg1_rel_dict.keys()))
    num_rel_triples1 = count_elm_first_triples(kg1_rel_dict)
    kg1_attr_dict = read_pred_first_triples(dataset_name, attr1_filename)
    num_attr1 = len(set(kg1_attr_dict.keys()))
    num_attr_triples1 = count_elm_first_triples(kg1_attr_dict)

    attr2_filename, rel2_filename = get_attr_and_rel_filenames(dataset_name, 'ent2')
    kg2_ent_out_dict, kg2_ent_in_dict = read_ent_first_rel_triples(dataset_name, rel2_filename)
    ent2_set = set(kg2_ent_out_dict.keys())
    ent2_in_set = set(kg2_ent_in_dict.keys())
    ent2_set.update(ent2_in_set)
    num_ent2 = len(ent2_set)
    kg2_rel_dict = read_pred_first_triples(dataset_name, rel2_filename)
    num_rel2 = len(set(kg2_rel_dict.keys()))
    num_rel_triples2 = count_elm_first_triples(kg2_rel_dict)
    kg2_attr_dict = read_pred_first_triples(dataset_name, attr2_filename)
    num_attr2 = len(set(kg2_attr_dict.keys()))
    num_attr_triples2 = count_elm_first_triples(kg2_attr_dict)

    print('Statistics for dataset %s are - '
          '\n num_ent1: %d, num_rel1: %d, num_attr1: %d, num_rel_triples1: %d, num_attr_triples1: %d,'
          '\n num_ent2: %d, num_rel2: %d, num_attr2: %d, num_rel_triples2: %d, num_attr_triples2: %d,'
          % (dataset_name, num_ent1, num_rel1, num_attr1, num_rel_triples1, num_attr_triples1,
             num_ent2, num_rel2, num_attr2, num_rel_triples2, num_attr_triples2))


def get_top_k_pred(dataset_name: str, file_name: str, k: int):
    pred_count_dict = {}
    pred_first_dict = read_pred_first_triples(dataset_name, file_name)
    for pred in pred_first_dict:
        count = len(pred_first_dict[pred])
        pred_count_dict[pred] = count

    sorted_pred_count_dict = sorted(pred_count_dict.items(), key=lambda x: x[1], reverse=True)
    top_k_pred = sorted_pred_count_dict[:k]
    return top_k_pred


def get_ent_degree_single_file(dataset_name: str, file_name: str, element_type: str,
                               total_degree_dict: dict, out_degree_dict: dict, in_degree_dict: dict):
    file_path = os.path.join(os.getcwd(), '..', 'dataset', dataset_name, file_name)
    if not os.path.exists(file_path):
        raise RuntimeError('Input file "%s" does not exist!' % file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            subj, pred, obj = line.strip().split('\t')
            if subj not in out_degree_dict:
                out_degree_dict[subj] = 1
            else:
                out_degree_dict[subj] += 1
            if subj not in total_degree_dict:
                total_degree_dict[subj] = 1
            else:
                total_degree_dict[subj] += 1
            if element_type == 'rel':
                if obj not in in_degree_dict:
                    in_degree_dict[obj] = 1
                else:
                    in_degree_dict[obj] += 1
                if obj not in total_degree_dict:
                    total_degree_dict[obj] = 1
                else:
                    total_degree_dict[obj] += 1

    return total_degree_dict, out_degree_dict, in_degree_dict


def get_most_k_degree_ent(dataset_name: str, k: int):
    ## processing ent1 ...
    ent1_total_degree_dict = {}
    ent1_attr_out_degree_dict = {}
    ent1_attr_in_degree_dict = {}
    ent1_rel_out_degree_dict = {}
    ent1_rel_in_degree_dict = {}
    attr1_filename, rel1_filename = get_attr_and_rel_filenames(dataset_name, 'ent1')
    ent1_total_degree_dict, ent1_attr_out_degree_dict, ent1_attr_in_degree_dict = \
        get_ent_degree_single_file(dataset_name, attr1_filename, 'attr',
                                   ent1_total_degree_dict, ent1_attr_out_degree_dict, ent1_attr_in_degree_dict)
    ent1_total_degree_dict, ent1_rel_out_degree_dict, ent1_rel_in_degree_dict = \
        get_ent_degree_single_file(dataset_name, rel1_filename, 'rel',
                                   ent1_total_degree_dict, ent1_rel_out_degree_dict, ent1_rel_in_degree_dict)
    top_k_ent1_total_degree = sorted(ent1_total_degree_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    print('top_k_ent1_total_degree ... ')
    print(top_k_ent1_total_degree)
    top_k_ent1_attr_out_degree = sorted(ent1_attr_out_degree_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    print('top_k_ent1_attr_out_degree ... ')
    print(top_k_ent1_attr_out_degree)
    top_k_ent1_rel_out_degree = sorted(ent1_rel_out_degree_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    print('top_k_ent1_rel_out_degree ... ')
    print(top_k_ent1_rel_out_degree)
    top_k_ent1_rel_in_degree = sorted(ent1_rel_in_degree_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    print('top_k_ent1_rel_in_degree ... ')
    print(top_k_ent1_rel_in_degree)
    ## processing ent2 ...
    ent2_total_degree_dict = {}
    ent2_attr_out_degree_dict = {}
    ent2_attr_in_degree_dict = {}
    ent2_rel_out_degree_dict = {}
    ent2_rel_in_degree_dict = {}
    attr2_filename, rel2_filename = get_attr_and_rel_filenames(dataset_name, 'ent2')
    ent2_total_degree_dict, ent2_attr_out_degree_dict, ent2_attr_in_degree_dict = \
        get_ent_degree_single_file(dataset_name, attr2_filename, 'attr',
                                   ent2_total_degree_dict, ent2_attr_out_degree_dict, ent2_attr_in_degree_dict)
    ent2_total_degree_dict, ent2_rel_out_degree_dict, ent2_rel_in_degree_dict = \
        get_ent_degree_single_file(dataset_name, rel2_filename, 'rel',
                                   ent2_total_degree_dict, ent2_rel_out_degree_dict, ent2_rel_in_degree_dict)
    top_k_ent2_total_degree = sorted(ent2_total_degree_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    print('top_k_ent2_total_degree ... ')
    print(top_k_ent2_total_degree)
    top_k_ent2_attr_out_degree = sorted(ent2_attr_out_degree_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    print('top_k_ent2_attr_out_degree ... ')
    print(top_k_ent2_attr_out_degree)
    top_k_ent2_rel_out_degree = sorted(ent2_rel_out_degree_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    print('top_k_ent2_rel_out_degree ... ')
    print(top_k_ent2_rel_out_degree)
    top_k_ent2_rel_in_degree = sorted(ent2_rel_in_degree_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    print('top_k_ent2_rel_in_degree ... ')
    print(top_k_ent2_rel_in_degree)

    return 0


def get_attr_and_rel_filenames(dataset_name: str, target_ent_list: str):
    if dataset_name in ['DBP15K_DE_EN_V1', 'DBP15K_FR_EN_V1', 'DBP100K_DE_EN_V1', 'DBP100K_FR_EN_V1',
                        'DW15K_V1', 'DY15K_V1']:
        if target_ent_list == 'ent1':
            attr_filename = 'attr_triples_1'
            rel_filename = 'rel_triples_1'
        elif target_ent_list == 'ent2':
            attr_filename = 'attr_triples_2'
            rel_filename = 'rel_triples_2'
        else:
            raise RuntimeError('Unknown target_ent_list: %s' % target_ent_list)
    elif dataset_name == 'DBP15K_FR_EN':
        if target_ent_list == 'ent1':
            attr_filename = 'fr_att_triples'
            rel_filename = 'fr_rel_triples'
        elif target_ent_list == 'ent2':
            attr_filename = 'en_att_triples'
            rel_filename = 'en_rel_triples'
        else:
            raise RuntimeError('Unknown target_ent_list: %s' % target_ent_list)
    elif dataset_name == 'DBP15K_JA_EN':
        if target_ent_list == 'ent1':
            attr_filename = 'ja_att_triples'
            rel_filename = 'ja_rel_triples'
        elif target_ent_list == 'ent2':
            attr_filename = 'en_att_triples'
            rel_filename = 'en_rel_triples'
        else:
            raise RuntimeError('Unknown target_ent_list: %s' % target_ent_list)
    elif dataset_name == 'DBP15K_ZH_EN':
        if target_ent_list == 'ent1':
            attr_filename = 'zh_att_triples'
            rel_filename = 'zh_rel_triples'
        elif target_ent_list == 'ent2':
            attr_filename = 'en_att_triples'
            rel_filename = 'en_rel_triples'
        else:
            raise RuntimeError('Unknown target_ent_list: %s' % target_ent_list)
    else:
        raise RuntimeError('Unknown dataset_name: %s' % dataset_name)

    return attr_filename, rel_filename


def merge_file(input_dir: str, output_file_name: str):
    merged_dict = {}
    process_dir = os.path.join(input_dir, 'temp')
    for file_name in os.listdir(process_dir):
        suffix = os.path.splitext(file_name)[-1]
        if suffix == '.json':
            with open(os.path.join(process_dir, file_name), 'r', encoding='utf-8') as f:
                cur_dict = json.load(f)
                merged_dict.update(cur_dict)

    output_dir = os.path.join(input_dir, 'processed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file_path = os.path.join(output_dir, output_file_name)
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(merged_dict, f, ensure_ascii=False, indent=4)
    print('File generated: %s' % output_file_path)

    return 0


def add_indent_to_json(input_file_path: str):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        input_dict = json.load(f)

    with open(input_file_path, 'w', encoding='utf-8') as f:
        json.dump(input_dict, f, ensure_ascii=False, indent=4)
    print('Output generated successfully: %s' % input_file_path)

    return 0


def analyze_ent_desc_dict(input_file_path: str):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        input_dict = json.load(f)
    num_total = len(input_dict)
    num_desc = 0
    num_error = 0
    num_none = 0
    for ent in input_dict:
        desc = input_dict[ent]
        if isinstance(desc, str):
            if desc == 'error':
                num_error += 1
            else:
                num_desc += 1
        elif isinstance(desc, type(None)):
            num_none += 1
        else:
            print('Unknown desc type: %s' % type(desc))

    print('num_desc: %d, num_none: %d, num_error: %d, num_total: %d' %
          (num_desc, num_none, num_error, num_total))

    return 0


def is_desc_valid(desc):
    if isinstance(desc, type(None)):
        return False
    elif isinstance(desc, str):
        if desc == 'error':
            return False
        else:
            return True
    else:
        raise RuntimeError('Unknown type for desc: %s' % desc)


def extract_dict_with_range(input_dict: dict, lb: int, ub: int):
    output_dict = {}

    target_keys = list(input_dict.keys())[lb:ub]
    for key in target_keys:
        output_dict[key] = input_dict[key]

    return output_dict


def split_with_mode(input_file_path: str, dataset_name: str, ea_data_mode: str):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        input_dict = json.load(f)

    if ea_data_mode == 'train-20':
        mode_lb = 0
        if '15K' in dataset_name:
            mode_ub = 3000
        elif '100K' in dataset_name:
            mode_ub = 20000
        else:
            raise RuntimeError('mode_ub calculation error with mode %s for dataset %s' % ea_data_mode)
    elif ea_data_mode == 'test-80':
        if '15K' in dataset_name:
            mode_lb = 3000
            mode_ub = 15000
        elif '100K' in dataset_name:
            mode_lb = 20000
            mode_ub = 100000
        else:
            raise RuntimeError('mode_ub calculation error with mode %s for dataset %s' % ea_data_mode)
    else:
        raise RuntimeError('Unknown ea_data_mode: %s' % ea_data_mode)

    output_dict = extract_dict_with_range(input_dict, mode_lb, mode_ub)
    output_file_path = input_file_path.replace('all', ea_data_mode)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)
    print('File generated: %s' % output_file_path)

    return 0


def get_kg_and_entLang(dataset_name: str, tar_ent_list_name: str):
    if dataset_name == 'DBP15K_ZH_EN':
        kg = 'dbpedia'
        if tar_ent_list_name == 'ent1':
            ent_lang = 'zh'
        elif tar_ent_list_name == 'ent2':
            ent_lang = 'en'
        else:
            raise RuntimeError('Unknown tar_ent_list_name: %s' % tar_ent_list_name)
    elif dataset_name == 'DBP15K_JA_EN':
        kg = 'dbpedia'
        if tar_ent_list_name == 'ent1':
            ent_lang = 'ja'
        elif tar_ent_list_name == 'ent2':
            ent_lang = 'en'
        else:
            raise RuntimeError('Unknown tar_ent_list_name: %s' % tar_ent_list_name)
    elif dataset_name == 'DBP15K_FR_EN':
        kg = 'dbpedia'
        if tar_ent_list_name == 'ent1':
            ent_lang = 'fr'
        elif tar_ent_list_name == 'ent2':
            ent_lang = 'en'
        else:
            raise RuntimeError('Unknown tar_ent_list_name: %s' % tar_ent_list_name)
    elif dataset_name == 'DBP15K_DE_EN_V1':
        kg = 'dbpedia'
        if tar_ent_list_name == 'ent1':
            ent_lang = 'en'
        elif tar_ent_list_name == 'ent2':
            ent_lang = 'de'
        else:
            raise RuntimeError('Unknown tar_ent_list_name: %s' % tar_ent_list_name)
    elif dataset_name == 'DW15K_V1':
        ent_lang = 'en'
        if tar_ent_list_name == 'ent1':
            kg = 'dbpedia'
        elif tar_ent_list_name == 'ent2':
            kg = 'wikidata'
        else:
            raise RuntimeError('Unknown tar_ent_list_name: %s' % tar_ent_list_name)
    elif dataset_name == 'DBP100K_FR_EN_V1':
        kg = 'dbpedia'
        if tar_ent_list_name == 'ent1':
            ent_lang = 'en'
        elif tar_ent_list_name == 'ent2':
            ent_lang = 'fr'
        else:
            raise RuntimeError('Unknown tar_ent_list_name: %s' % tar_ent_list_name)
    else:
        raise RuntimeError('Unknown dataset_name: %s' % dataset_name)

    return kg, ent_lang


def compute_llm_output_distribution(dataset_name: str, llm_mode: str, ea_data_mode: str, triple_sel: str,
                                    exp_sel: str, tar_sel: str, ptm_model: str, llm_type: str, llm_temp: str,
                                    max_rep: int, lower_bound: int, upper_bound: int, train_percentage: float):
    print('Computing llm output distribution ... %s=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d' %
          (dataset_name, llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel, ptm_model,
           llm_type, llm_temp, max_rep, lower_bound, upper_bound))

    tar_splits = tar_sel.split('-')
    tar_strategy = tar_splits[0]
    num_class = int(tar_splits[1])

    batch_size = 100
    ## check if debugging mode
    if upper_bound == 100:
        ptm_sim_upper_bound = 3000
    else:
        ptm_sim_upper_bound = upper_bound

    if ea_data_mode == 'train-20':
        train_bound = int(ptm_sim_upper_bound * train_percentage)
    elif ea_data_mode == 'test-80':
        train_bound = int(ptm_sim_upper_bound * train_percentage)
    else:
        raise RuntimeError('Unhandled ea_data_mode: %s for ea_by_llm' % ea_data_mode)

    ptm_sim_dict_file_name = ('ptm_sim_dict=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.json' %
                              (llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel, ptm_model,
                               llm_type, llm_temp, max_rep, lower_bound, ptm_sim_upper_bound))

    ptm_sim_dict_dir = os.path.join(os.getcwd(), '..', 'output', 'ptm_sim_dict', dataset_name)
    if not os.path.exists(ptm_sim_dict_dir):
        os.makedirs(ptm_sim_dict_dir)
    ptm_sim_dict_path = os.path.join(ptm_sim_dict_dir, ptm_sim_dict_file_name)
    if os.path.exists(ptm_sim_dict_path):
        with open(ptm_sim_dict_path, 'r', encoding='utf-8') as f:
            print('Loading ptm_sim_dict ... %s' % ptm_sim_dict_path)
            ptm_sim_dict = json.load(f)
    else:
        raise RuntimeError('ptm_sim_dict not found: %s' % ptm_sim_dict_path)

    ## read llm output and add to distribution
    llm_output_dir = os.path.join(os.getcwd(), '..', 'output', 'ea_result', dataset_name, 'processed')
    prefix = '%s=%s=%s=%s=%s=%s=%s=%s' % (llm_mode, ea_data_mode, tar_sel,
                                          exp_sel, triple_sel, ptm_model, llm_type, llm_temp)
    suffix = '=%d=%d.json' % (lower_bound, upper_bound)
    llm_output_distribution_dict = {}
    flag = False
    for file_name in os.listdir(llm_output_dir):
        if file_name == prefix + suffix:
            continue
        if file_name.startswith(prefix) and file_name.endswith(suffix):
            file_path = os.path.join(llm_output_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                llm_output_dict = json.load(f)
                ## initialize llm_output_distribution_dict with ptm_sim_dict
                if not flag:
                    for ent1 in llm_output_dict:
                        ent2_sim_pair_list = ptm_sim_dict[ent1]
                        sorted_ent2_sim_pair_list = sorted(ent2_sim_pair_list, key=lambda x: x[1], reverse=True)
                        for ent2_sim_pair in sorted_ent2_sim_pair_list[:num_class]:
                            ent2 = ent2_sim_pair[0]
                            if ent1 not in llm_output_distribution_dict:
                                llm_output_distribution_dict[ent1] = [[ent2, 0]]
                            else:
                                llm_output_distribution_dict[ent1].append([ent2, 0])
                    flag = True
                ## add single file output to llm_output_distribution_dict
                for ent1 in tqdm(llm_output_dict):
                    llm_ent2 = llm_output_dict[ent1]
                    ## remove [] from llm_output if it exists
                    if llm_ent2.startswith('[') and llm_ent2.endswith(']'):
                        llm_ent2 = llm_ent2[1:-1]
                    ent2_pair_list = llm_output_distribution_dict[ent1]
                    for index in range(len(ent2_pair_list)):
                        ent2 = ent2_pair_list[index][0]
                        if llm_ent2 == ent2:
                            llm_output_distribution_dict[ent1][index][1] += 1
                            break

    llm_output_distribution_dir = os.path.join(os.getcwd(), '..', 'output', 'ea_result', dataset_name, 'distribution')
    if not os.path.exists(llm_output_distribution_dir):
        os.makedirs(llm_output_distribution_dir)

    if ea_data_mode.startswith('train'):
        ## generate train distribution
        train_bound = int(upper_bound * train_percentage)
        train_distribution_dict = {k: v for k, v in list(llm_output_distribution_dict.items())[:train_bound]}
        train_distribution_file_name = 'train_distribution=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.json' % \
                                       (llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel,
                                        ptm_model, llm_type, llm_temp, max_rep, lower_bound, train_bound)
        train_distribution_file_path = os.path.join(llm_output_distribution_dir, train_distribution_file_name)
        if len(train_distribution_dict) > 0:
            ## save llm_output_distribution_dict as file
            with open(train_distribution_file_path, 'w', encoding='utf-8') as f:
                json.dump(train_distribution_dict, f, ensure_ascii=False, indent=4)
            print('Output generated successfully: %s' % train_distribution_file_path)
        else:
            print('No output distribution: %s' % train_distribution_file_path)
        ## generate test distribution
        test_distribution_dict = {k: v for k, v in list(llm_output_distribution_dict.items())[train_bound:]}
        test_distribution_file_name = 'test_distribution=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.json' % \
                                      (llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel,
                                       ptm_model, llm_type, llm_temp, max_rep, train_bound, upper_bound)
        test_distribution_file_path = os.path.join(llm_output_distribution_dir, test_distribution_file_name)
        if len(test_distribution_dict) > 0:
            ## save llm_output_distribution_dict as file
            with open(test_distribution_file_path, 'w', encoding='utf-8') as f:
                json.dump(test_distribution_dict, f, ensure_ascii=False, indent=4)
            print('Output generated successfully: %s' % test_distribution_file_path)
        else:
            print('No output distribution: %s' % test_distribution_file_path)
    elif ea_data_mode.startswith('test'):
        test_distribution_file_name = 'test_distribution=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.json' % \
                                      (llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel,
                                       ptm_model, llm_type, llm_temp, max_rep, lower_bound, upper_bound)
        test_distribution_file_path = os.path.join(llm_output_distribution_dir, test_distribution_file_name)
        if len(llm_output_distribution_dict) > 0:
            ## save llm_output_distribution_dict as file
            with open(test_distribution_file_path, 'w', encoding='utf-8') as f:
                json.dump(llm_output_distribution_dict, f, ensure_ascii=False, indent=4)
            print('Output generated successfully: %s' % test_distribution_file_path)
        else:
            print('No output distribution: %s' % test_distribution_file_path)

    return 0


def get_desc_dicts(dataset_name: str, ea_data_mode: str, desc_gen_model: str):
    ent_desc_dir = os.path.join(os.getcwd(), '..', 'output', 'ent_desc_dict', dataset_name)
    if desc_gen_model in ['gpt3.5', 'baichuan2']:
        ent1_desc_file_name = 'ent_desc_dict_by_llm=%s=%s=ent1.json' % \
                              (ea_data_mode, desc_gen_model)
        ent2_desc_file_name = 'ent_desc_dict_by_llm=%s=%s=ent2.json' % \
                              (ea_data_mode, desc_gen_model)
    elif desc_gen_model == 'sparql':
        ent1_desc_file_name = 'ent_desc_dict_by_retrieval=%s=ent1.json' % \
                              (ea_data_mode)
        ent2_desc_file_name = 'ent_desc_dict_by_retrieval=%s=ent2.json' % \
                              (ea_data_mode)
    else:
        raise RuntimeError('Unknown desc_model: %s' % desc_gen_model)

    ent1_desc_file_path = os.path.join(ent_desc_dir, ent1_desc_file_name)
    with open(ent1_desc_file_path, 'r', encoding='utf-8') as f:
        ent1_desc_dict = json.load(f)
    ent2_desc_file_path = os.path.join(ent_desc_dir, ent2_desc_file_name)
    with open(ent2_desc_file_path, 'r', encoding='utf-8') as f:
        ent2_desc_dict = json.load(f)

    ## replace empty desc with ent iri
    for ent in ent1_desc_dict:
        if not is_desc_valid(ent1_desc_dict[ent]):
            ent1_desc_dict[ent] = ent
    for ent in ent2_desc_dict:
        if not is_desc_valid(ent2_desc_dict[ent]):
            ent2_desc_dict[ent] = ent

    return ent1_desc_dict, ent2_desc_dict


def get_triple_dicts(dataset_name: str, ea_data_mode: str, num_triple: int, triple_strategy: str):
    ent_triple_dir = os.path.join(os.getcwd(), '..', 'output', 'ent_triple_dict', dataset_name)
    ent1_triple_file_name = 'ent1_triple=%s=%d=%s.json' % (ea_data_mode, num_triple, triple_strategy)
    ent2_triple_file_name = 'ent2_triple=%s=%d=%s.json' % (ea_data_mode, num_triple, triple_strategy)

    ent1_triple_file_path = os.path.join(ent_triple_dir, ent1_triple_file_name)
    with open(ent1_triple_file_path, 'r', encoding='utf-8') as f:
        ent1_triple_dict = json.load(f)
    ent2_triple_file_path = os.path.join(ent_triple_dir, ent2_triple_file_name)
    with open(ent2_triple_file_path, 'r', encoding='utf-8') as f:
        ent2_triple_dict = json.load(f)

    return ent1_triple_dict, ent2_triple_dict


def remove_dataset_from_file_name(dir: str, dataset_name: str):
    for file_name in os.listdir(dir):
        if file_name.startswith(dataset_name) or 'pmm' in file_name:
            old_file_path = os.path.join(dir, file_name)
            new_file_name = file_name.replace('pmm', 'mpnet')
            if file_name.startswith(dataset_name):
                new_file_name = file_name[len(dataset_name) + 1:]
            new_file_path = os.path.join(dir, new_file_name)
            # rename files
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{file_name}' to '{new_file_name}'")

            file_name.startswith(dataset_name)

    return 0


def rename_llm_mode(directory: str):
    for file_name in os.listdir(directory):
        if file_name.endswith('.json') or file_name.endswith('.xlsx') or file_name.endswith('.pth'):
            if ('ea_zs_llm' in file_name) or ('ea_icl_llm' in file_name):
                old_file_path = os.path.join(directory, file_name)
                if 'ea_zs_llm' in file_name:
                    new_file_name = file_name.replace('ea_zs_llm', 'zs')
                elif 'ea_icl_llm' in file_name:
                    new_file_name = file_name.replace('ea_icl_llm', 'icl')
                else:
                    raise RuntimeError('Unknown error for llm_mode')

                new_file_path = os.path.join(directory, new_file_name)
                # rename files
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{file_name}' to '{new_file_name}'")

                file_name.startswith(dataset_name)

    return 0


def move_to_processed_dir(dataset_name: str, llm_mode: str, ea_data_mode: str, triple_sel: str,
                          exp_sel: str, tar_sel: str, ptm_model_name: str, llm_type: str, llm_temp: str):
    source_dir = os.path.join(os.getcwd(), '..', 'output', 'ea_result', dataset_name)
    target_dir = os.path.join(source_dir, 'processed')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get a list of all .json files in the current directory.
    # This is done by listing all files in the source directory and filtering for those with a '.json' extension.
    ea_output_file_name_prefix = '%s=%s=%s=%s=%s=%s=%s=%s' % \
                                 (llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel,
                                  ptm_model_name, llm_type, llm_temp)

    json_files = [f for f in os.listdir(source_dir) if f.startswith(ea_output_file_name_prefix)]
    # Iterate over each .json file in the list.
    for json_file in json_files:
        # Construct the full path to the source file.
        source_path = os.path.join(source_dir, json_file)

        # Construct the full path to the target file.
        # The target file will have the same name as the source file but will be located in the target directory.
        target_path = os.path.join(target_dir, json_file)

        # Use the shutil.move() function to move the file from the source path to the target path.
        # This will effectively rename and/or move the file to the new location.
        shutil.move(source_path, target_path)

    # Print a message indicating the completion of the file transfer.
    print(f"Moved all .json files to {target_dir}")


def select_top_k_from_ptm_sim_dict(k, dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel, tar_sel,
                                   ptm_model_name, llm_type, llm_temp, max_rep, lower_bound, upper_bound):
    ptm_sim_dict = {}
    ptm_sim_dict_file_name = ('ptm_sim_dict=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.json' %
                              (llm_mode, ea_data_mode, triple_sel, exp_sel, tar_sel, ptm_model_name,
                               llm_type, llm_temp, max_rep, lower_bound, upper_bound))

    ptm_sim_dict_dir = os.path.join(os.getcwd(), '..', 'output', 'ptm_sim_dict', dataset_name)
    if not os.path.exists(ptm_sim_dict_dir):
        os.makedirs(ptm_sim_dict_dir)
    ptm_sim_dict_path = os.path.join(ptm_sim_dict_dir, ptm_sim_dict_file_name)

    ## if ptm_sim_dict exists, load it
    if os.path.exists(ptm_sim_dict_path):
        with open(ptm_sim_dict_path, 'r', encoding='utf-8') as f:
            print('Loading ptm_sim_dict ... %s' % ptm_sim_dict_path)
            ptm_sim_dict = json.load(f)

    ## select top k
    for ent1 in ptm_sim_dict:
        ent2_sim_pair_list = ptm_sim_dict[ent1]
        ptm_sim_dict[ent1] = ent2_sim_pair_list[:k]

    ## Save the similarity dictionary
    with open(ptm_sim_dict_path, 'w', encoding='utf-8') as f:
        json.dump(ptm_sim_dict, f, ensure_ascii=False, indent=4)
    print('Slimmed the ptm_sim_dict successfully: %s' % ptm_sim_dict_path)

    return 0


if __name__ == '__main__':
    #### split_with_mode
    ## ['DBP15K_FR_EN', 'DBP15K_JA_EN', 'DBP15K_ZH_EN', 'DW15K_V1', 'DY15K_V1',
    # 'DBP15K_DE_EN_V1', 'DBP15K_FR_EN_V1', 'DBP100K_DE_EN_V1', 'DBP100K_FR_EN_V1']
    # dataset_name = 'DW15K_V1'
    # target_ent_list = ['ent1', 'ent2']
    # ea_data_mode_list = ['train-20', 'test-80']
    # for target_ent in target_ent_list:
    #     for ea_data_mode in ea_data_mode_list:
    #         input_file_path = os.path.join(os.getcwd(), '..', 'output', 'ent_triple_dict', dataset_name,
    #                                        '%s_triple=all=5=rand.json' % target_ent)
    #         split_with_mode(input_file_path, dataset_name, ea_data_mode)

    # # ['DBP15K_ZH_EN', 'DBP15K_JA_EN', 'DBP15K_FR_EN', 'DW15K_V1', 'DY15K_V1',
    # # 'DBP15K_DE_EN_V1', 'DBP15K_FR_EN_V1', 'DBP100K_DE_EN_V1', 'DBP100K_FR_EN_V1']
    dataset_name = 'DY15K_V1'
    # compute_dataset_statistics(dataset_name)
    get_most_k_degree_ent(dataset_name, k=5)