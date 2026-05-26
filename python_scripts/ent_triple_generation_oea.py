import json
import os
import argparse
from tqdm import tqdm
import re  # 引入 re 模块

from llm_classification import get_attr_and_rel_filenames, get_random_samples_from_dict
from overall_process import parse_dataset_name, parse_ea_data_mode
from utility import read_groundtruth_with_mode, get_element_triples, get_frequent_rel_and_attr, \
    read_ent_first_attr_triples, read_ent_first_rel_triples, split_with_mode

def get_short_name(uri_string: str) -> str:
    """
    一个更强大的URI缩短函数，能处理DBpedia和Freebase的格式。
    - 移除首尾的尖括号 <>
    - 对于HTTP URI，返回最后一个'/'之后的部分。
    - 对于非URI（如 /m/01m4kpp），原样返回。
    """
    if not isinstance(uri_string, str):
        return uri_string

    clean_uri = uri_string.strip()

    # 1. 移除尖括号
    if clean_uri.startswith('<') and clean_uri.endswith('>'):
        clean_uri = clean_uri[1:-1]

    # 2. 如果是HTTP URI，则分割并取最后一部分
    if clean_uri.startswith('http://') or clean_uri.startswith('https://'):
        # Freebase predicate: http://rdf.freebase.com/ns/people.person.date_of_birth
        # DBpedia predicate: http://dbpedia.org/ontology/birthDate
        return clean_uri.split('/')[-1]

    # 3. 如果不是HTTP URI (例如 '/m/01m4kpp')，则原样返回
    return clean_uri

def generate_freq_attr_rel_triple(ent_list: list, attr_triples: dict, rel_out_triples: dict, rel_in_triples: dict,
                                  ent_sorted_attr_list: list, ent_sorted_rel_out_list: list,
                                  ent_sorted_rel_in_list: list, num_triple: int, dataset_name: str):
    ent_triple_dict = {}
    print('Generating frequent ent triples ... ')
    for ent in tqdm(ent_list):
        ## add attribute triples
        triple_direction = 'out'
        element_type = 'attr'
        ent_triple_dict = add_single_element_triple(
            ent_triple_dict, ent, attr_triples, ent_sorted_attr_list, element_type, triple_direction, num_triple, dataset_name)
        ## add out relation triples
        element_type = 'rel'
        ent_triple_dict = add_single_element_triple(
            ent_triple_dict, ent, rel_out_triples, ent_sorted_rel_out_list,
            element_type, triple_direction, num_triple, dataset_name)
        ## add in relation triples
        triple_direction = 'in'
        ent_triple_dict = add_single_element_triple(
            ent_triple_dict, ent, rel_in_triples, ent_sorted_rel_in_list,
            element_type, triple_direction, num_triple, dataset_name)

    return ent_triple_dict

def generate_rand_attr_rel_triple(tar_ent_list: list, num_triple: int,
                                  attr_dict: dict, rel_out_dict: dict, rel_in_dict: dict,
                                  dataset_name: str):
    ent_triple_dict = {}
    print('Generating random triples ... ')
    for tar_ent in tqdm(tar_ent_list):
        ent_name = get_short_name(tar_ent)
        if tar_ent in attr_dict.keys():
            ent_attr_dict = attr_dict[tar_ent]
            if num_triple < len(ent_attr_dict):
                ent_attr_dict = get_random_samples_from_dict(ent_attr_dict, num_triple)
            for attr in ent_attr_dict:
                attr_name = get_short_name(attr)
                value = ent_attr_dict[attr][0]
                ent_triple_text = '(%s, %s, %s)' % (ent_name, attr_name, value)
                if tar_ent not in ent_triple_dict:
                    ent_triple_dict[tar_ent] = [ent_triple_text]
                else:
                    ent_triple_dict[tar_ent].append(ent_triple_text)
        if tar_ent in rel_out_dict.keys():
            ent_rel_out_dict = rel_out_dict[tar_ent]
            if num_triple < len(ent_rel_out_dict):
                ent_rel_out_dict = get_random_samples_from_dict(ent_rel_out_dict, num_triple)
            for rel_out in ent_rel_out_dict:
                rel_out_name = get_short_name(rel_out)
                object_val = ent_rel_out_dict[rel_out][0]
                object_name = get_short_name(object_val)
                ent_triple_text = '(%s, %s, %s)' % (ent_name, rel_out_name, object_name)
                if tar_ent not in ent_triple_dict:
                    ent_triple_dict[tar_ent] = [ent_triple_text]
                else:
                    ent_triple_dict[tar_ent].append(ent_triple_text)
        if tar_ent in rel_in_dict.keys():
            ent_rel_in_dict = rel_in_dict[tar_ent]
            if num_triple < len(ent_rel_in_dict):
                ent_rel_in_dict = get_random_samples_from_dict(ent_rel_in_dict, num_triple)
            for rel_in in ent_rel_in_dict:
                rel_in_name = get_short_name(rel_in)
                suject = ent_rel_in_dict[rel_in][0]
                suject_name = get_short_name(suject)
                ent_triple_text = '(%s, %s, %s)' % (suject_name, rel_in_name, ent_name)
                if tar_ent not in ent_triple_dict:
                    ent_triple_dict[tar_ent] = [ent_triple_text]
                else:
                    ent_triple_dict[tar_ent].append(ent_triple_text)

    return ent_triple_dict

def add_single_element_triple(ent_triple_dict: dict, ent: str, element_triples: dict,
                              ent_sorted_element_list: list, element_type: str, triple_direction: str,
                              num_triple: int, dataset_name: str):
    if ent in element_triples:
        ent_element_dict = element_triples[ent]
        triple_index = 0
        for element_index in range(len(ent_sorted_element_list)):
            cur_element = ent_sorted_element_list[element_index]
            if cur_element in ent_element_dict:
                if element_type == 'attr':
                    so = ent_element_dict[cur_element][0]
                    
                    # --- 修改开始：清洗数据 ---
                    # 1. 去除 RDF 类型后缀 (例如 ^^<http://...>)
                    if '^^' in so:
                        so = so.split('^^')[0]
                    
                    # 2. 去除多余的引号 (包括转义引号 \" 和普通引号 ")
                    so = so.replace('\"', '').replace('"', '')
                    # --- 修改结束 ---

                elif element_type == 'rel':
                    so = get_short_name(ent_element_dict[cur_element][0])
                else:
                    raise RuntimeError('Unknown element_type: %s' % element_type)

                ent_name = get_short_name(ent)
                element_name = get_short_name(cur_element)
                
                # 注意：这里保持原来的逻辑，但 so 已经被清洗过了
                if triple_direction == 'out':
                    triple = '(%s, %s, %s)' % (ent_name.replace("/",'.'), (element_name.replace("/",'.')).split(".")[-1], so.replace("/",'.'))
                elif triple_direction == 'in':
                    triple = '(%s, %s, %s)' % (so.replace("/",'.'), (element_name.replace("/",'.')).split(".")[-1], ent_name.replace("/",'.'))
                else:
                    raise RuntimeError('Unknown triple_direction: %s' % triple_direction)
                
                if ent not in ent_triple_dict:
                    ent_triple_dict[ent] = [triple]
                else:
                    ent_triple_dict[ent].append(triple)
                triple_index += 1
                if triple_index >= num_triple:
                    break

    return ent_triple_dict

def generate_ent_triple(dataset_name: str, ea_data_mode: str, num_triple: int, triple_strategy: str):
    # complete dataset_name and ea_data_mode
    dataset_name = parse_dataset_name(dataset_name)
    ea_data_mode = parse_ea_data_mode(ea_data_mode)
    print(dataset_name)
    gt_dict = read_groundtruth_with_mode(dataset_name, ea_data_mode)
    ent1_list = list(gt_dict.keys())
    ent2_list = list(gt_dict.values())

    attr_triples1, rel_out_triples1, rel_in_triples1, attr_triples2, rel_out_triples2, rel_in_triples2 = \
        get_element_triples(dataset_name)

    if triple_strategy == 'freq':
        # print(12345)
        ent1_sorted_attr_list, ent1_sorted_rel_out_list, ent1_sorted_rel_in_list, \
            ent2_sorted_attr_list, ent2_sorted_rel_out_list, ent2_sorted_rel_in_list = \
            get_frequent_rel_and_attr(dataset_name, ea_data_mode)

        ent1_triple_dict = generate_freq_attr_rel_triple(
            ent1_list, attr_triples1, rel_out_triples1, rel_in_triples1,
            ent1_sorted_attr_list, ent1_sorted_rel_out_list, ent1_sorted_rel_in_list, num_triple, dataset_name)
        # print()
        ent2_triple_dict = generate_freq_attr_rel_triple(
            ent2_list, attr_triples2, rel_out_triples2, rel_in_triples2,
            ent2_sorted_attr_list, ent2_sorted_rel_out_list, ent2_sorted_rel_in_list, num_triple, dataset_name)
        # print(ent2_triple_dict)
    elif triple_strategy == 'rand':
        ent1_triple_dict = generate_rand_attr_rel_triple(ent1_list, num_triple,
                                                         attr_triples1, rel_out_triples1, rel_in_triples1, dataset_name)
        ent2_triple_dict = generate_rand_attr_rel_triple(ent2_list, num_triple,
                                                         attr_triples2, rel_out_triples2, rel_in_triples2, dataset_name)
    else:
        raise RuntimeError('Unknown tar_strategy: %s' % triple_strategy)

    output_dir = os.path.join(os.getcwd(), '..', 'output', 'ent_triple_dict', dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ent1_triple_file_name = 'ent1_triple=%s=%d=%s.json' % (ea_data_mode, num_triple, triple_strategy)
    ent1_triple_file_path = os.path.join(output_dir, ent1_triple_file_name)
    with open(ent1_triple_file_path, 'w', encoding='utf-8') as f:
        json.dump(ent1_triple_dict, f, ensure_ascii=False, indent=4)
        print('Generated file ... %s' % ent1_triple_file_path)
    split_with_mode(ent1_triple_file_path, dataset_name, 'train-20')
    split_with_mode(ent1_triple_file_path, dataset_name, 'test-80')

    ent2_triple_file_name = 'ent2_triple=%s=%d=%s.json' % (ea_data_mode, num_triple, triple_strategy)
    ent2_triple_file_path = os.path.join(output_dir, ent2_triple_file_name)
    with open(ent2_triple_file_path, 'w', encoding='utf-8') as f:
        json.dump(ent2_triple_dict, f, ensure_ascii=False, indent=4)
        print('Generated file ... %s' % ent2_triple_file_path)
    split_with_mode(ent2_triple_file_path, dataset_name, 'train-20')
    split_with_mode(ent2_triple_file_path, dataset_name, 'test-80')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TRE triples')
    parser.add_argument('-d', '--dataset_name', type=str, required=True,
                        help='dze, dje, dfe, dw, dy, ddev, dfev, ddev100k, dfev100k,fbdb,fbdb_link')
    parser.add_argument('--ea_data_mode', type=str, default='all', help='all, train, test')
    parser.add_argument('--num_triple', type=int, default=5, help='5, 3, 7')
    parser.add_argument('--triple_strategy', type=str, default='freq', help='freq, rand')
    args = parser.parse_args()

    generate_ent_triple(args.dataset_name, args.ea_data_mode, args.num_triple, args.triple_strategy)