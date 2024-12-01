import rrc_evaluation_funcs
import numpy as np 
from collections import namedtuple
import math
import csv
import json
import zipfile
import os

# 将ppocr格式转化为icdar格式
def ppocrToIcdar(ppocrPath, outputDir):
    with open(ppocrPath,'r',encoding='utf-8')as fp:
        s = [i[:-1].split('\t') for i in fp.readlines()]
        counter = 1
        for i in enumerate(s):
            # 解析标注内容，需要import json
            anno = json.loads(i[1][1])
            # 通过规则筛选出文件名
            filename = i[1][0][:-4]
            # 有的电表有表号，有的没有，需要逐一遍历
            with open(os.path.join(outputDir, '{filename}'.format(filename= 'res_img_' + str(counter)) + '.txt'), 'w', encoding='utf-8') as f:
                for j in range(len(anno)): 
                    label = anno[j-1]['transcription']
                    # xmin, xmax, ymin, ymax的计算逻辑
                    x1 = min(int(anno[j-1]['points'][0][0]),int(anno[j-1]['points'][1][0]),int(anno[j-1]['points'][2][0]),int(anno[j-1]['points'][3][0]))
                    x2 = max(int(anno[j-1]['points'][0][0]),int(anno[j-1]['points'][1][0]),int(anno[j-1]['points'][2][0]),int(anno[j-1]['points'][3][0]))
                    y1 = min(int(anno[j-1]['points'][0][1]),int(anno[j-1]['points'][1][1]),int(anno[j-1]['points'][2][1]),int(anno[j-1]['points'][3][1]))
                    y2 = max(int(anno[j-1]['points'][0][1]),int(anno[j-1]['points'][1][1]),int(anno[j-1]['points'][2][1]),int(anno[j-1]['points'][3][1]))
                    f.write('{x1},{y1}, {x2}, {y2}, {label} \n'.format(x1=x1, y1=y1, x2=x2, y2=y2, label=label))
                

# 将文件夹转换为zip文件
def zipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')
 
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()



def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """
 
    # for module, alias in evaluation_imports().items():
    #     globals()[alias] = importlib.import_module(module)
 
    def one_to_one_match(row, col):
        cont = 0
        for j in range(len(recallMat[0])):  # len(detRects)
            if recallMat[row, j] >= evaluationParams['AREA_RECALL_CONSTRAINT'] and precisionMat[row, j] >= \
                    evaluationParams['AREA_PRECISION_CONSTRAINT']:  # 0.8,0.4
                cont = cont + 1
        if cont != 1:
            return False
        cont = 0
        for i in range(len(recallMat)):
            if recallMat[i, col] >= evaluationParams['AREA_RECALL_CONSTRAINT'] and precisionMat[i, col] >= \
                    evaluationParams['AREA_PRECISION_CONSTRAINT']:
                cont = cont + 1
        if cont != 1:
            return False
 
        if recallMat[row, col] >= evaluationParams['AREA_RECALL_CONSTRAINT'] and precisionMat[row, col] >= \
                evaluationParams['AREA_PRECISION_CONSTRAINT']:
            return True
        return False
 
    def one_to_many_match(gtNum):
        many_sum = 0
        detRects = []
        for detNum in range(len(recallMat[0])):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and detNum not in detDontCareRectsNum:
                if precisionMat[gtNum, detNum] >= evaluationParams['AREA_PRECISION_CONSTRAINT']:  # 0.4
                    many_sum += recallMat[gtNum, detNum]
                    detRects.append(detNum)
        if many_sum >= evaluationParams['AREA_RECALL_CONSTRAINT']:  # 0.8
            return True, detRects
        else:
            return False, []
 
    def many_to_one_match(detNum):
        many_sum = 0
        gtRects = []
        for gtNum in range(len(recallMat)):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCareRectsNum:
                if recallMat[gtNum, detNum] >= evaluationParams['AREA_RECALL_CONSTRAINT']:  # 0.8
                    many_sum += precisionMat[gtNum, detNum]
                    gtRects.append(gtNum)
        if many_sum >= evaluationParams['AREA_PRECISION_CONSTRAINT']:  # 0.4
            return True, gtRects
        else:
            return False, []
 
    def area(a, b):
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin) + 1
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin) + 1
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        else:
            return 0.
 
    def center(r):
        x = float(r.xmin) + float(r.xmax - r.xmin + 1) / 2.
        y = float(r.ymin) + float(r.ymax - r.ymin + 1) / 2.
        return Point(x, y)
 
    def point_distance(r1, r2):
        distx = math.fabs(r1.x - r2.x)
        disty = math.fabs(r1.y - r2.y)
        return math.sqrt(distx * distx + disty * disty)
 
    def center_distance(r1, r2):
        return point_distance(center(r1), center(r2))
 
    def diag(r):
        w = (r.xmax - r.xmin + 1)
        h = (r.ymax - r.ymin + 1)
        return math.sqrt(h * h + w * w)
 
    perSampleMetrics = {}
 
    methodRecallSum = 0
    methodPrecisionSum = 0
 
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    Point = namedtuple('Point', 'x y')
 
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath, evaluationParams['GT_SAMPLE_NAME_2_ID'])  # 'gt_img_([0-9]+).txt'
    subm = rrc_evaluation_funcs.load_zip_file(submFilePath, evaluationParams['DET_SAMPLE_NAME_2_ID'], True)  # 'res_img_([0-9]+).txt'
 
    numGt = 0
    numDet = 0
 
    for resFile in gt:  # 一张ground truth
        # 189, b'121,0,177,12,###\r\n90,110,528,193,hellmann\r\n89,197,269,251,parcel\r\n295,204,528,254,systems\r\n'
        gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])
        # 121,0,177,12,###
        # 90,110,528,193,hellmann
        # 89,197,269,251,parcel
        # 295,204,528,254,systems
        recall = 0
        precision = 0
        hmean = 0
        recallAccum = 0.
        precisionAccum = 0.
        gtRects = []
        detRects = []
        gtPolPoints = []
        detPolPoints = []
        gtDontCareRectsNum = []  # Array of Ground Truth Rectangles' keys marked as don't Care
        detDontCareRectsNum = []  # Array of Detected Rectangles' matched with a don't Care GT
        pairs = []
        evaluationLog = ""
 
        recallMat = np.empty([1, 1])
        precisionMat = np.empty([1, 1])
 
        # be careful with the parameters here, make sure withTranscription is set to True if your files contain transcriptions
        pointsList, _, transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(gtFile,
                                                                                                       evaluationParams[
                                                                                                           'CRLF'],
                                                                                                       # False
                                                                                                       True, True,
                                                                                                       False) 
         
        for n in range(len(pointsList)):  # 一张ground truth上的一个文本实例
            points = pointsList[n]  # list, [121.0, 0.0, 177.0, 12.0]
            transcription = transcriptionsList[n]
            dontCare = transcription == "###"  # 如果transcription是"###"，则标为dontCare
            gtRect = Rectangle(*points)  # Rectangle(xmin=121.0, ymin=0.0, xmax=177.0, ymax=12.0)
            gtRects.append(gtRect)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCareRectsNum.append(len(gtRects) - 1)  # 一张图中dontcare文本实例的索引
 
        evaluationLog += "GT rectangles: " + str(len(gtRects)) + (
            " (" + str(len(gtDontCareRectsNum)) + " don't care)\n" if len(gtDontCareRectsNum) > 0 else "\n")
        # GT rectangles: 4 (1 don't care)
 
        if resFile in subm:  # 对应这张ground truth的识别结果
            detFile = rrc_evaluation_funcs.decode_utf8(subm[resFile])
            # 一张图片的检测结果
            # <class 'str'>
            # 121,0,177,12
            # 90,110,528,193
            # 89,197,269,251
            # 295,204,528,254
            pointsList, _, _ = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(detFile,
                                                                                          evaluationParams['CRLF'],
                                                                                          True, True, False)
            # [[121.0, 0.0, 177.0, 12.0], [90.0, 110.0, 528.0, 193.0], [89.0, 197.0, 269.0, 251.0], [295.0, 204.0, 528.0, 254.0]]
            for n in range(len(pointsList)):  # 这张识别结果中的一个文本实例
                points = pointsList[n]
                detRect = Rectangle(*points)
                detRects.append(detRect)
                detPolPoints.append(points)
                if len(gtDontCareRectsNum) > 0:
                    for dontCareRectNum in gtDontCareRectsNum:  # 遍历对应gt中的don't care文本实例
                        dontCareRect = gtRects[dontCareRectNum]
                        intersected_area = area(dontCareRect, detRect)  # 拿这张图识别结果中的一个文本实例和对应gt中所有标为don't care的计算intersection area
                        rdDimensions = ((detRect.xmax - detRect.xmin + 1) * (detRect.ymax - detRect.ymin + 1)) #计算出detRect的面积
                        if rdDimensions == 0:
                            precision = 0
                        else:
                            precision = intersected_area / rdDimensions # 计算出precision
                        if precision > evaluationParams['AREA_PRECISION_CONSTRAINT']:  # 阈值为0.4
                            detDontCareRectsNum.append(len(detRects) - 1)
                            # 识别结果中的一个文本实例和gt中标为don't care的一个文本实例匹配上了，记录这个识别的文本实例的索引
                            break
                            # 可能还与其它的don't care实例也匹配，但不关心了，break
 
            evaluationLog += "DET rectangles: " + str(len(detRects)) + (
                " (" + str(len(detDontCareRectsNum)) + " don't care)\n" if len(detDontCareRectsNum) > 0 else "\n")
            # DET rectangles: 4 (1 don't care)
 
            if len(gtRects) == 0:  # 这张图上没有gt box,dont't care也没有
                recall = 1
                precision = 0 if len(detRects) > 0 else 1
 
            if len(detRects) > 0:
                # Calculate recall and precision matrices
                outputShape = [len(gtRects), len(detRects)]  # 一行是1个gt，一列是1个det
                recallMat = np.empty(outputShape)  # 记录了一张gt中的每个文本实例与对应det中的每个文本实例的recall
                precisionMat = np.empty(outputShape)  # 记录了一张gt中的每个文本实例与对应det中的每个文本实例的precision
                gtRectMat = np.zeros(len(gtRects), np.int8)  # 记录每个gt是否匹配，匹配则值为1
                detRectMat = np.zeros(len(detRects), np.int8)  # 记录每个det是否匹配，匹配则值为1
                # 分别遍历gt和det中的每一个text instance
                for gtNum in range(len(gtRects)):
                    for detNum in range(len(detRects)):
                        # nested for loops make sure detection boxes are matched with ground truth boxes
                        rG = gtRects[gtNum]
                        rD = detRects[detNum]
                        intersected_area = area(rG, rD)
                        rgDimensions = ((rG.xmax - rG.xmin + 1) * (rG.ymax - rG.ymin + 1))
                        rdDimensions = ((rD.xmax - rD.xmin + 1) * (rD.ymax - rD.ymin + 1))
                        recallMat[gtNum, detNum] = 0 if rgDimensions == 0 else intersected_area / rgDimensions
                        # 召回率: 一个gt和一个det的intersection面积与该gt面积的比例
                        precisionMat[gtNum, detNum] = 0 if rdDimensions == 0 else intersected_area / rdDimensions
                        # 准确率: 一个gt和一个det的intersection面积与该det面积的比例
 
                # 在接下来判断一对一、一对多、多对一三种情况时，都忽略don't care的gt和det
                # Find one-to-one matches
                evaluationLog += "Find one-to-one matches\n"
                for gtNum in range(len(gtRects)):
                    for detNum in range(len(detRects)):
                        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCareRectsNum and detNum not in detDontCareRectsNum:
                            match = one_to_one_match(gtNum, detNum)
                            # 一对一，即1个gt只与1个det满足recall和precision的阈值，同时这个det也只与这个gt满足recall和precision的阈值
                            # recallMat的一行代表一个gt和所有det的recall，一列代表一个det和所有gt的recall，precisionMat也是如此
                            # 当前gt所在的行与当前det所在的列中只有交叉点即该gt与det的recall和precision大于设定的阈值，即1对1匹配
                            if match is True:
                                rG = gtRects[gtNum]
                                rD = detRects[detNum]
                                normDist = center_distance(rG, rD)
                                normDist /= diag(rG) + diag(rD)
                                normDist *= 2.0
                                if normDist < evaluationParams['EV_PARAM_IND_CENTER_DIFF_THR']:  # 1
                                    '''
                                    DIoU
                                    用对角线距离把检测框和预测框的中心点距离进行归一化,在IOU值相同的情况下,
                                    两个框的中心点归一化距离越小,代表预测框与目标框的对比效果越好。当DIoU值为1时,说明两个框无重合
                                    优点是可以直接最小化两个目标框的距离,比GIOU收敛的更快。
                                    '''
                                    gtRectMat[gtNum] = 1  # 当前gt已匹配，后面判断一对多和多对一的情况时，跳过该gt
                                    detRectMat[detNum] = 1  # 当前det已匹配，后面判断一对多和多对一的情况时，跳过该det
                                    recallAccum += evaluationParams['MTYPE_OO_O']  # 1
                                    precisionAccum += evaluationParams['MTYPE_OO_O']
                                    pairs.append({'gt': gtNum, 'det': detNum, 'type': 'OO'})
                                    evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"
                                else:
                                    evaluationLog += "Match Discarded GT #" + str(gtNum) + " with Det #" + str(
                                        detNum) + " normDist: " + str(normDist) + " \n"
                # Find one-to-many matches
                evaluationLog += "Find one-to-many matches\n"
                for gtNum in range(len(gtRects)):
                    if gtNum not in gtDontCareRectsNum:
                        match, matchesDet = one_to_many_match(gtNum)
                        # 一对多，即1个gt对应多个det，遍历1个gt所在的行，若有det与该gt的precision大于设定阈值，记录该det以及gt与该det的recall
                        # 若所有满足precision的det与gt的recall和大于设定阈值，则该gt与所有记录的det匹配上了
                        if match is True:
                            gtRectMat[gtNum] = 1
                            recallAccum += evaluationParams['MTYPE_OM_O']  # 0.8
                            '''
                            为什么有时候recall和precision加1有时候加0.8, 可以认为是对不同匹配结果的惩罚
                            因为ICDAR2013的ground truth已经是word level的,所以衡量算法对one-to-many(检测结果将一个word分成了好几个bbox)
                            的惩罚要大于many-to-one(将多个word用一个bbox框出来)。
                            '''
                            precisionAccum += evaluationParams['MTYPE_OM_O'] * len(matchesDet)
                            pairs.append({'gt': gtNum, 'det': matchesDet, 'type': 'OM'})
                            for detNum in matchesDet:
                                detRectMat[detNum] = 1
                            evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(matchesDet) + "\n"
 
                # Find many-to-one matches
                evaluationLog += "Find many-to-one matches\n"
                for detNum in range(len(detRects)):
                    if detNum not in detDontCareRectsNum:
                        match, matchesGt = many_to_one_match(detNum)
                        # 多对一，即多个gt对应1个det，遍历1个det所在的列，若有gt与该det的recall大于设定阈值，记录该gt以及det与该gt的precision
                        # 若所有满足recall的gt与该det的precision和大于设定阈值，则该det与所有记录的gt匹配上了
                        if match is True:
                            detRectMat[detNum] = 1
                            recallAccum += evaluationParams['MTYPE_OM_M'] * len(matchesGt)  # 1
                            precisionAccum += evaluationParams['MTYPE_OM_M']
                            pairs.append({'gt': matchesGt, 'det': detNum, 'type': 'MO'})
                            for gtNum in matchesGt:
                                gtRectMat[gtNum] = 1
                            evaluationLog += "Match GT #" + str(matchesGt) + " with Det #" + str(detNum) + "\n"
 
                numGtCare = (len(gtRects) - len(gtDontCareRectsNum))
                if numGtCare == 0:
                    recall = float(1)
                    precision = float(0) if len(detRects) > 0 else float(1)
                else:
                    recall = float(recallAccum) / numGtCare
                    precision = float(0) if (len(detRects) - len(detDontCareRectsNum)) == 0 else float(
                        precisionAccum) / (len(detRects) - len(detDontCareRectsNum))
                hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
 
        evaluationLog += "Recall = " + str(recall) + "\n"
        evaluationLog += "Precision = " + str(precision) + "\n"
 
        methodRecallSum += recallAccum
        methodPrecisionSum += precisionAccum
        numGt += len(gtRects) - len(gtDontCareRectsNum)
        numDet += len(detRects) - len(detDontCareRectsNum)
 
        perSampleMetrics[resFile] = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'recallMat': [] if len(detRects) > 100 else recallMat.tolist(),
            'precisionMat': [] if len(detRects) > 100 else precisionMat.tolist(),
            'gtPolPoints': gtPolPoints,
            'detPolPoints': detPolPoints,
            'gtDontCare': gtDontCareRectsNum,
            'detDontCare': detDontCareRectsNum,
            'evaluationParams': evaluationParams,
            'evaluationLog': evaluationLog
        }
 
    methodRecall = 0 if numGt == 0 else methodRecallSum / numGt
    methodPrecision = 0 if numDet == 0 else methodPrecisionSum / numDet
    methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (
            methodRecall + methodPrecision)
 
    methodMetrics = {'precision': methodPrecision, 'recall': methodRecall, 'hmean': methodHmean}
 
    resDict = {'calculated': True, 'Message': '', 'method': methodMetrics, 'per_sample': perSampleMetrics}
    
    print('precision for the entire dataset:', resDict['method']['precision'])
    print('recall for the entire dataset:', resDict['method']['recall'])
    print('hmean for the entire dataset:', resDict['method']['hmean'])
    
    return resDict


if __name__ == '__main__':
    ppocrPredictioniPath = './images_prediction.txt'
    icdarPredictionDir = './prediction'

    ppocrGTPath = './images__gt.txt'
    icdarGTDir = './gt'

    groundtruthZipPath = './gt.zip'
    predictionZipPath = './submit.zip'

    default_param = {
            'AREA_RECALL_CONSTRAINT': 0.8,
            'AREA_PRECISION_CONSTRAINT': 0.4,
            'EV_PARAM_IND_CENTER_DIFF_THR': 1,
            'MTYPE_OO_O': 1.,
            'MTYPE_OM_O': 0.8,
            'MTYPE_OM_M': 1.,
            'GT_SAMPLE_NAME_2_ID': 'gt_img_([0-9]+).txt',
            'DET_SAMPLE_NAME_2_ID': 'res_img_([0-9]+).txt',
            'CRLF': False  # Lines are delimited by Windows CRLF format
        }

    
    ppocrToIcdar(ppocrPredictioniPath, icdarPredictionDir)
    ppocrToIcdar(ppocrGTPath, icdarGTDir)
    zipDir(icdarPredictionDir, predictionZipPath)
    zipDir(icdarGTDir, groundtruthZipPath)

    resDict = evaluate_method(groundtruthZipPath, predictionZipPath, default_param)

    # 把结果导出到csv文件
    with open('output.csv', 'w', newline='') as csvfile:
        fieldnames = ['file_id', 'precision', 'recall', 'hmean']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id, metrics in resDict['per_sample'].items():
            writer.writerow({'file_id': sample_id, 'precision': metrics['precision'], 'recall': metrics['recall'], 'hmean': metrics['hmean']})
