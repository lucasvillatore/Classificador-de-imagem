import argparse
from os import listdir
from os.path import isfile, join
import re
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

files_base_regex = '[0-9]{2}-scale_[0-9]_im_1_col.png'
files_extra_regex = '[0-9]{2}-scale_[0-9]_im_[2-9]_col.png'

def getArguments():
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Caminho das imagens", default="./")
    return parser.parse_args()

def getFilesBase(path, regex):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    if ('lab5.py' in onlyfiles):
        onlyfiles.remove('lab5.py')
    regex = re.compile(regex)
    onlyImages = [i for i in onlyfiles if regex.match(i)]
    onlyImages.sort()

    return onlyImages

def equalize_hist(img):
    for c in range(0, 2):
        img[:,:,c] = cv.equalizeHist(img[:,:,c])

    return img

def getTrainingData(files, path):
    arrayTraining = []
    for i in files:
        arrayTraining.append(getData(i, path))

    return arrayTraining

def getTestingData(files, path):
    for i in files:
        pass

def getArrayTrainingDataAndLabels(data_extra):

    arrayTrainingData = []
    arrayTrainingLabels = []

    for i in data_extra:
        arrayTrainingData.append(i['histogram'])
        arrayTrainingLabels.append(i['class'])

    return [arrayTrainingData, arrayTrainingLabels]

def getClass(string):
    classNumber, text = string.split("-", 2)

    return classNumber

def getData(fileName, path):
    base_path = arguments.path + "/" + fileName
    file = cv.imread(base_path)

    hist_range = [0, 256]
    hist_size = [256]
    
    rgb_planes = cv.split(file)

    b_hist = cv.calcHist(rgb_planes, [0], None, hist_size, hist_range)
    g_hist = cv.calcHist(rgb_planes, [1], None, hist_size, hist_range)
    r_hist = cv.calcHist(rgb_planes, [2], None, hist_size, hist_range)

    classNumber, text = fileName.split("-", 2)
    text = text.split("_", 4)
    scale = int(text[1])
    number = int(text[3])
    data = {
        "class" : int(str(classNumber) +  str(scale) + str(number)),
        "name" : fileName,
        "histogram" : [
            np.argmax(b_hist),
            np.argmax(g_hist),
            np.argmax(r_hist)
        ]
    }
    return data

def getFileName(distance, result):
    scale_number = int(result) % 100
    scale = str(scale_number // 10)
    number = str(scale_number % 10)
    className = str(int(result)).replace(str(scale_number), '')
        
    file = className + '-scale_'+scale+'_im_'+number+'_col.png'
    return file, className
    
if __name__ == '__main__':
    arguments   = getArguments()

    files_extra = getFilesBase(arguments.path, files_extra_regex)  
    data_extra = getTrainingData(files_extra, arguments.path)
    

    arrayTrainingData, arrayTrainingLabels = getArrayTrainingDataAndLabels(data_extra)
    knn = cv.ml.KNearest_create()
    knn.train(np.array(arrayTrainingData, dtype=np.float32), cv.ml.ROW_SAMPLE, np.array(arrayTrainingLabels, dtype=np.float32))
    
    files_base  = getFilesBase(arguments.path, files_base_regex)
    hit = 0
    for i in files_base:
        dataTest = getData(i, arguments.path)
        ret, result, neighbours, distance = knn.findNearest(np.array([dataTest['histogram']], dtype=np.float32), 1)
        distance = distance[0][0]
        result = result[0][0]
        file, classResult = getFileName(distance,result)
        print('{} {} {}'.format(i,file, distance/100))
        if (int(getClass(i)) == int(classResult)):
            hit += 1

    print('Imagens classificadas corretamente {} de {}'.format(hit, len(files_base)))