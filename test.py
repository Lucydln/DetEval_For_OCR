import json

gt_dict = {}
det_dict = {}

def load_data(filepath, isGT = False, isDET = False):    
    with open(filepath,'r',encoding='utf-8')as fp:
        s = [i[:-1].split('\t') for i in fp.readlines()]
        for i in enumerate(s):
            # 解析标注内容，需要import json
            anno = json.loads(i[1][1])
            # 通过规则筛选出文件名
            filename = i[1][0][:-4]
            if isGT:
                gt_dict[filename] = [[], []]
            if isDET:
                det_dict[filename] = [[], []]
            # 有的电表有表号，有的没有，需要逐一遍历
            for j in range(len(anno)): 
                label = anno[j-1]['transcription']
                # xmin, xmax, ymin, ymax的计算逻辑
                x1 = min(int(anno[j-1]['points'][0][0]),int(anno[j-1]['points'][1][0]),int(anno[j-1]['points'][2][0]),int(anno[j-1]['points'][3][0]))
                x2 = max(int(anno[j-1]['points'][0][0]),int(anno[j-1]['points'][1][0]),int(anno[j-1]['points'][2][0]),int(anno[j-1]['points'][3][0]))
                y1 = min(int(anno[j-1]['points'][0][1]),int(anno[j-1]['points'][1][1]),int(anno[j-1]['points'][2][1]),int(anno[j-1]['points'][3][1]))
                y2 = max(int(anno[j-1]['points'][0][1]),int(anno[j-1]['points'][1][1]),int(anno[j-1]['points'][2][1]),int(anno[j-1]['points'][3][1]))
                if isGT:
                    gt_dict[filename][0].append([x1, x2, y1, y2])
                    gt_dict[filename][1].append(label)
                if isDET:
                    det_dict[filename][0].append([x1, x2, y1, y2])
                    det_dict[filename][1].append(label)


load_data(r'c:\Users\dingln51075\Desktop\Intern\labelme\images_with_label.txt', True)
