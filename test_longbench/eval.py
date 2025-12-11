import os, json

files = os.listdir('results')
# === 修改 1: 增加 Avg_CoT 列标题 ===
output = ["Model\tOverall\tEasy\tHard\tShort\tMedium\tLong\tAvg_CoT"]
compensated = False

for file in files:
    filename = os.path.join('results', file)
    try:
        pred_data = json.load(open(filename, encoding='utf-8'))
    except Exception as e:
        pred_data = [json.loads(line) for line in open(filename, encoding='utf-8')]
    
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    
    # === 修改 2: 初始化 CoT 总长度变量 ===
    total_cot_length = 0

    for pred in pred_data:
        acc = int(pred['judge'])
        if compensated and pred["pred"] == None:
            acc = 0.25
        
        # === 修改 3: 累加 cot_length，使用 .get 默认值为 0 以防旧数据缺失该字段 ===
        total_cot_length += pred.get('cot_length', 0)

        if pred["difficulty"] == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        if pred['length'] == "short":
            short += 1
            short_acc += acc
        elif pred['length'] == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc

    name = '.'.join(file.split('.')[:-1])
    
    # === 修改 4: 计算平均 CoT 长度 (处理分母为0的情况) ===
    avg_cot = round(total_cot_length / len(pred_data), 1) if len(pred_data) > 0 else 0.0

    # === 修改 5: 将 avg_cot 添加到输出字符串的末尾 ===
    output.append(
        name + '\t' + 
        str(round(100*(easy_acc+hard_acc)/len(pred_data), 1)) + '\t' + 
        str(round(100*easy_acc/easy, 1) if easy > 0 else 0) + '\t' + 
        str(round(100*hard_acc/hard, 1) if hard > 0 else 0) + '\t' + 
        str(round(100*short_acc/short, 1) if short > 0 else 0) + '\t' + 
        str(round(100*medium_acc/medium, 1) if medium > 0 else 0) + '\t' + 
        str(round(100*long_acc/long, 1) if long > 0 else 0) + '\t' + 
        str(avg_cot)
    )

open('result.txt', 'w', encoding='utf-8').write('\n'.join(output))