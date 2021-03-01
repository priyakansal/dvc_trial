# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 09:40:00 2021

@author: priyakansal
"""
import argparse
import sys
import time
from datetime import datetime
import os




parser = argparse.ArgumentParser(description = 'evaluates VCX')
parser.add_argument('--vcxpath', type = str, default= None, help = 'path to the VCX repository' )
parser.add_argument('--use_tflite_runtime', type=int, default=0)
parser.add_argument('--use_tensorRT', type=int, default=0)
parser.add_argument('--image_path', type=str, default="officeVideo1.mp4")
#parser.add_argument('--save_path', type=str, default="results")

args = parser.parse_args()
# get the current dir
cdir = os.getcwd() 

#vcxlitepath = args.vcxlitepath
vcxpath = args.vcxpath
os.chdir(vcxpath) # Navigate
sys.path.insert(1, vcxpath)


if args.use_tflite_runtime ==1 and  args.use_tensorRT==1:
    raise Exception('Tf Lite and TensorRT can not be true at a time, Please Change the input')

if args.use_tensorRT == 1: 
    import cv2
    from person_detection_tensorRT import model_intialize, recognize_combined_features
    global Detector, encoder,head_pose1,head_pose2,head_pose3,AG_Model,face_features
    st = time.time()
    tracker, encoder,head_pose1,head_pose2, head_pose3,AG_Model,face_features,EM_Model,MD_Model,GE_Model,FK_Model,Detector= model_intialize()
    et = time.time()
    ModelInitializeTime = et-st
    
elif args.use_tflite_runtime==1: 
    import cv2
    import tflite_runtime.interpreter as tflite
    from combi_test import combi, model_intialize
    st = time.time()
    interpreters = {}
    for mp in ["person_face.tflite","person_tracking.tflite","age_gender.tflite","mask.tflite",\
            "headpose_capsule.tflite","headpose_nos.tflite","headpose_var.tflite","facialkeypoint.tflite",\
            "gaze.tflite"]:
    
        interpreter = tflite.Interpreter(model_path="models/" + mp)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreters[mp.split(".")[0]] = [interpreter,input_details,output_details]
    # initialize
    image_for_intialize = cv2.imread(os.path.join(cdir, args.image_path, 'video_img_00000.png'))
    model_intialize(interpreters,image_for_intialize)
    et = time.time()
    ModelInitializeTime = et-st    

else:
    from person_detection import model_intialize, recognize_combined_features, config_variable
    global Detector, encoder, head_pose1, head_pose2, head_pose3, AG_Model, face_features
    
    st = time.time()
    tracker, Detector, encoder, head_pose1, head_pose2, head_pose3, AG_Model, face_features, EM_Model, MD_Model = model_intialize()
    #tracker,Detector, encoder,Face_Detector,head_pose1,head_pose2, head_pose3,AG_Model =  model_intialize()   # for Jetson Nano
    et = time.time()
    ModelInitializeTime = et-st


os.chdir(cdir)# Navigate back to the current directory ##########################


import json
import re
import pandas as pd

from pascalvoc import *
from validateinputs import *
evaluator = Evaluator()
print('evaluator loaded')



def recognize(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    start_time = time.time()
    results = recognize_combined_features(data, tracker, Detector, encoder, head_pose1, head_pose2, head_pose3, AG_Model, face_features, EM_Model, MD_Model)
#    results = recognize_combined_features(data,tracker,Detector, encoder,Face_Detector,head_pose1,head_pose2, head_pose3,AG_Model) #JetsonNono
    end_time = time.time()
    f.close()
    return results, end_time-start_time

def recognize_trt(filename):
    image = cv2.imread(filename)
    start_time = time.time()
    results,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_= recognize_combined_features(image,tracker, encoder,head_pose1,head_pose2, head_pose3,AG_Model,face_features,EM_Model,MD_Model,GE_Model,FK_Model,Detector)
    end_time = time.time()
    return results, end_time-start_time



def createtfliteresults(image_path, save_path, interpreters):
    
    '''
    Save the .txt files for every image in the following format (minmaxnormalized coord):
        0 prob x y r b  
        1 prob x y r b
        labels are:
        0 - person
        1 - face
    '''
    path = image_path
    xs = [img for img in os.listdir(path) if img.endswith(".png")]
    images = sorted(xs, key = lambda x: (int(re.sub('\D', '', x)), x)) 
    
    # avoiding_large_tracking_delay_for_specific_AWS_instances
    
    _,_ = combi(cv2.imread(os.path.join(path, images[0])),interpreters,preparation=1)
    _,_ =  combi(cv2.imread(os.path.join(path, images[0])),interpreters,preparation=1)
    _,_ =  combi(cv2.imread(os.path.join(path, images[0])),interpreters,preparation=1)
    
    speed = []
    save_path = os.path.join(save_path, 'detection') 
    if not os.path.isdir(save_path):
            os.makedirs(save_path)
    count = 0
    final = []
    names = []
    for im in images: 
        st1 = time.time()
        f_gt= open(os.path.join(save_path, (im[:-4]+ ".txt")),"w+")
        filename = os.path.join(path,im)
        image = cv2.imread(filename)
#        print(im)
        results, _ = combi(image,interpreters)
        et1 = time.time()
        speed.append(et1-st1)
#        print(et1-st1)
#        results = json.loads(results)
        if len(results['objects'])!=0:
#            print(results)
#            print('=='*70)
            for i in range(len(results['objects'])):
                if len(results['objects'][i])>4:
                    f_gt.write('0 {} {} {} {} {} \n'.format(     # '0' - person, '1' - 'face'
                            results['objects'][i]['prob'],
                            float(results['objects'][i]['pos'][0]), 
                            float(results['objects'][i]['pos'][1]), 
                            float(results['objects'][i]['pos'][2]), 
                            float(results['objects'][i]['pos'][3])))
                    f_gt.write('1 {} {} {} {} {} \n'.format(
                            results['objects'][i]['face']['prob'],
                            float(results['objects'][i]['face']['pos'][0]), 
                            float(results['objects'][i]['face']['pos'][1]), 
                            float(results['objects'][i]['face']['pos'][2]), 
                            float(results['objects'][i]['face']['pos'][3])))
                elif len(results['objects'][i])== 4:
                    f_gt.write('0 {} {} {} {} {} \n'.format(
                            results['objects'][i]['prob'],
                            float(results['objects'][i]['pos'][0]), 
                            float(results['objects'][i]['pos'][1]), 
                            float(results['objects'][i]['pos'][2]), 
                            float(results['objects'][i]['pos'][3])))
                
        else:
            f_gt.write('\n')
#            print(im)
            count = count+1
                
        f_gt.close()
        final.append(results)
        names.append(im)
    sub = pd.DataFrame([i for i in range(len(names))])
    sub['names'] = names
    sub['pred'] = final
#    sub.to_csv('D:/EvaluateVCX/predtable.csv')
#    print('result files are saved at %s' %save_path )
    return sub, count, sum(speed)/len(images), len(images)

def createresults(image_path, save_path):
    
    '''
    Save the .txt files for every image in the following format (minmaxnormalized coord):
        0 prob x y r b  
        1 prob x y r b
        labels are:
        0 - person
        1 - face
    '''
    
    path = image_path
    xs = [img for img in os.listdir(path) if img.endswith(".png")]
    images = sorted(xs, key = lambda x: (int(re.sub('\D', '', x)), x)) 
    speed = []
    save_path = os.path.join(save_path, 'detection') 
    if not os.path.isdir(save_path):
            os.makedirs(save_path)
    count = 0
#    _, _ = recognize(os.path.join(path, images[0]))
#    _, _ = recognize(os.path.join(path, images[0]))
#    _, _ = recognize(os.path.join(path, images[0]))
    final = []
    names = []
    for im in images: 
       
        f_gt= open(os.path.join(save_path, (im[:-4]+ ".txt")),"w+")
        filename = os.path.join(path,im)
#        print(im)
        results, time = recognize(filename)
        speed.append(time)
        results = json.loads(results)
        if len(results['objects'])!=0:
            for i in range(len(results['objects'])):
                if len(results['objects'][i])>4:
                    f_gt.write('0 {} {} {} {} {} \n'.format(     # '0' - person, '1' - 'face'
                            results['objects'][i]['prob'],
                            float(results['objects'][i]['pos'][0]), 
                            float(results['objects'][i]['pos'][1]), 
                            float(results['objects'][i]['pos'][2]), 
                            float(results['objects'][i]['pos'][3])))
                    f_gt.write('1 {} {} {} {} {} \n'.format(
                            results['objects'][i]['face']['prob'],
                            float(results['objects'][i]['face']['pos'][0]), 
                            float(results['objects'][i]['face']['pos'][1]), 
                            float(results['objects'][i]['face']['pos'][2]), 
                            float(results['objects'][i]['face']['pos'][3])))
                elif len(results['objects'][i])== 4:
                    f_gt.write('0 {} {} {} {} {} \n'.format(
                            results['objects'][i]['prob'],
                            float(results['objects'][i]['pos'][0]), 
                            float(results['objects'][i]['pos'][1]), 
                            float(results['objects'][i]['pos'][2]), 
                            float(results['objects'][i]['pos'][3])))
                
        else:
            f_gt.write('\n')
#            print(im)
            count = count+1
                
        f_gt.close()
        final.append(results)
        names.append(im)
    sub = pd.DataFrame([i for i in range(len(names))])
    sub['names'] = names
    sub['pred'] = final
#    sub.to_csv('D:/EvaluateVCX/predtable.csv')
#    print('result files are saved %s' %save_path )
    return sub, count, sum(speed)/len(images), len(images)
    
def createtrtresults(image_path, save_path):
    
    '''
    Save the .txt files for every image in the following format (minmaxnormalized coord):
        0 prob x y r b  
        1 prob x y r b
        labels are:
        0 - person
        1 - face
    '''
    
    path = image_path
    xs = [img for img in os.listdir(path) if img.endswith(".png")]
    images = sorted(xs, key = lambda x: (int(re.sub('\D', '', x)), x)) 
    speed = []
    save_path = os.path.join(save_path, 'detection') 
    if not os.path.isdir(save_path):
            os.makedirs(save_path)
    count = 0
#    _, _ = recognize(os.path.join(path, images[0]))
#    _, _ = recognize(os.path.join(path, images[0]))
#    _, _ = recognize(os.path.join(path, images[0]))
    final = []
    names = []
    for im in images: 
       
        f_gt= open(os.path.join(save_path, (im[:-4]+ ".txt")),"w+")
        filename = os.path.join(path,im)
#        print(im)
        results, time = recognize_trt(filename)
        speed.append(time)
        results = json.loads(results)
        if len(results['objects'])!=0:
            for i in range(len(results['objects'])):
                if len(results['objects'][i])>4:
                    f_gt.write('0 {} {} {} {} {} \n'.format(     # '0' - person, '1' - 'face'
                            results['objects'][i]['prob'],
                            float(results['objects'][i]['pos'][0]), 
                            float(results['objects'][i]['pos'][1]), 
                            float(results['objects'][i]['pos'][2]), 
                            float(results['objects'][i]['pos'][3])))
                    f_gt.write('1 {} {} {} {} {} \n'.format(
                            results['objects'][i]['face']['prob'],
                            float(results['objects'][i]['face']['pos'][0]), 
                            float(results['objects'][i]['face']['pos'][1]), 
                            float(results['objects'][i]['face']['pos'][2]), 
                            float(results['objects'][i]['face']['pos'][3])))
                elif len(results['objects'][i])== 4:
                    f_gt.write('0 {} {} {} {} {} \n'.format(
                            results['objects'][i]['prob'],
                            float(results['objects'][i]['pos'][0]), 
                            float(results['objects'][i]['pos'][1]), 
                            float(results['objects'][i]['pos'][2]), 
                            float(results['objects'][i]['pos'][3])))
                
        else:
            f_gt.write('\n')
#            print(im)
            count = count+1
                
        f_gt.close()
        final.append(results)
        names.append(im)
    sub = pd.DataFrame([i for i in range(len(names))])
    sub['names'] = names
    sub['pred'] = final
#    sub.to_csv('D:/EvaluateVCX/predtable.csv')
#    print('result files are saved %s' %save_path )
    return sub, count, sum(speed)/len(images), len(images)

def calculate_map(current_path, save_path, IOU_threshold):
    gt_folder,gt_format, gt_coord_type, img_size, det_folder, det_format, det_coord_type = get_validated(current_path, save_path)
    
    all_bounding_boxes, all_classes = read_bounding_boxes(gt_folder,
                                                      True,
                                                      gt_format,
                                                      gt_coord_type,
                                                      img_size=img_size)
    # Get detected boxes
    all_bounding_boxes, all_classes = read_bounding_boxes(det_folder,
                                                          False,
                                                          det_format,
                                                          det_coord_type,
                                                          all_bounding_boxes,
                                                          all_classes,
                                                          img_size=img_size)
    
    
    detections = evaluator.plot_precision_recall_curve(
    all_bounding_boxes,  # Object containing all bounding boxes (ground truths and detections)
    IOU_threshold=0.5,  # IOU threshold
    method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
    show_AP= False,  # Show Average Precision in the title of the plot
    show_interpolated_precision=False,  # Don't plot the interpolated precision curve
    save_path=save_path,
    show_graphic=False)
    
    acc_AP = 0
    count_validated_classes = 0
    results = {}
    # Each detection is a class
    for metrics_per_class in detections:
#        result ={}
        # Get metric values per each class
        cl = metrics_per_class['class']
        if cl == '0':
            cl = 'person'
        else:
            cl = 'face'
        ap = metrics_per_class['AP']
        precision = metrics_per_class['precision']
        recall = metrics_per_class['recall']
        total_positives = metrics_per_class['total positives']
        total_TP = metrics_per_class['total TP']
        total_FP = metrics_per_class['total FP']
        
    
        if total_positives > 0:
            count_validated_classes += 1
            acc_AP = acc_AP + ap
            prec = ['%.2f' % p for p in precision]
            rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)
            print('AP: %s (%s)' % (ap_str, cl))
            results[cl] = ap_str
        else:
            results[cl] = 0
#        results.append(result)
    mAP = acc_AP / count_validated_classes
    mAP_str = "{0:.2f}%".format(mAP * 100)
    results['mAP'] = mAP_str
    
    print('mAP: %s' % mAP_str)
    
    return results



def produce_results(vcxpath, image_path, use_tflite_runtime):
    current_path = os.path.dirname(os.path.abspath("__file__"))
    gt = pd.read_csv(os.path.join(current_path, 'gt.csv')).groundtruth.values.tolist()
#    save_path = os.path.join(current_path, 'results',  datetime.now().strftime("%Y%m%d%H%M%S"))
    save_path = os.path.join(current_path, 'results')
#    if os.path.exists(save_path):
#        save_path = os.path.join(save_path, 'latest')
    os.makedirs(save_path, exist_ok = True)
        
    image_path = os.path.join(current_path ,image_path)
    if use_tflite_runtime == 1:
        sub, noResultsFrame, performanceSpeedPerFrame, evaluatedExampleCount = createtfliteresults(image_path, save_path, interpreters)
    elif args.use_tensorRT == 1:    
        sub, noResultsFrame, performanceSpeedPerFrame, evaluatedExampleCount = createresults(image_path, save_path)
    else:    
        sub, noResultsFrame, performanceSpeedPerFrame, evaluatedExampleCount = createresults(image_path, save_path)
    sub['groundtruth'] = gt
    sub.to_csv(os.path.join(save_path, 'results.csv'))
    r = {}
    r['test_version'] = os.path.split(vcxpath)[-1]
    r['modelInitializeTime'] = ModelInitializeTime
    r['evaluatedExampleCount'] = evaluatedExampleCount
    
    
    em = calculate_map(current_path, save_path, IOU_threshold=0.5)
    evalmet = {}
    evalmet['personDetectionAccuracy (AP)'] = em['person']
    evalmet['faceDetectionAccuracy (AP)'] = em['face']
    evalmet['meanAccuracy (mAP)'] = em['mAP']
    evalmet['performanceSpeedPerFrame'] = performanceSpeedPerFrame
    evalmet['noResultsFrame'] = noResultsFrame
    
    r['evaluationMetrics'] = evalmet
    
    results = {'results':r}
    detection_path = os.path.join(save_path, 'detection')
    with open(os.path.join(save_path + '/result.txt'), 'w+') as outfile:
        json.dump(r, outfile)
    body = json.dumps(results)
    outfile.close()
    
    print(body)
    print('==================== All results are saved in %s ====================' %save_path )
    



    
produce_results(args.vcxpath, args.image_path, args.use_tflite_runtime) 
   
    
