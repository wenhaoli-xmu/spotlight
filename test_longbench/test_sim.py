import argparse
import os
from rouge_score.rouge_scorer import RougeScorer
import json
import numpy as np
import tqdm


ALL_DATA_LIST = [
    "narrativeqa.jsonl",
    "qasper.jsonl", 
    "multifieldqa_en.jsonl", 
    "multifieldqa_zh.jsonl", 
    "hotpotqa.jsonl", 
    "2wikimqa.jsonl",
    "musique.jsonl",
    "dureader.jsonl",
    "gov_report.jsonl", 
    "qmsum.jsonl", 
    "multi_news.jsonl", 
    "vcsum.jsonl", 
    "trec.jsonl", 
    "triviaqa.jsonl", 
    "samsum.jsonl", 
    "lsht.jsonl",
    "passage_count.jsonl", 
    "passage_retrieval_en.jsonl", 
    "passage_retrieval_zh.jsonl", 
    "lcc.jsonl", 
    "repobench-p.jsonl"
]


def get_file_list(dir):
    assert os.path.exists(dir), f"{dir} dost not exists!"
    lst = list(filter(lambda x: isinstance(x, str) and x.endswith(".jsonl"), os.listdir(dir)))
    if 'result.jsonl' in lst:
        lst.remove('result.jsonl')
    return lst


def sort_lst(lst):
    data_order = {}
    for i, data in enumerate(ALL_DATA_LIST):
        data_order[data] = i
    return sorted(lst, key=lambda x: data_order[x])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir1', type=str)
    parser.add_argument('dir2', type=str)
    parser.add_argument('--metric', type=str, default='rougeL')
    args = parser.parse_args()

    lst1 = get_file_list(args.dir1)
    lst2 = get_file_list(args.dir2)
    lst = set(lst1).intersection(set(lst2))
    lst = sort_lst(lst)

    scorer = RougeScorer([args.metric], use_stemmer=True)
    result = {}

    for filename in tqdm.tqdm(lst):
        path1 = os.path.join(args.dir1, filename)
        path2 = os.path.join(args.dir2, filename) 

        with open(path1, 'r') as f1:
            with open(path2, 'r') as f2:
                lines1 = f1.readlines()
                lines2 = f2.readlines()

                if len(lines1) != len(lines2):
                    continue

                scores = []
                for line1, line2 in zip(lines1, lines2):

                    if args.dir1 == args.dir2:
                        assert line1 == line2

                    line1 = json.loads(line1)
                    line2 = json.loads(line2)
                    score1 = scorer.score(line1['pred'], line2['pred'])[args.metric].precision
                    score2 = scorer.score(line2['pred'], line1['pred'])[args.metric].precision
                    scores.append((score1 + score2) / 2)                      

                average = np.mean(scores)

                result[filename] = average
    
    
    for key, value in result.items():
        print(f"{key:30} {value}")
    print(f"average: {np.mean(list(result.values()))}")
