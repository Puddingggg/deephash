import pandas as pd
import tensorflow as tf
import numpy as np
import csv
hashcode = pd.read_csv("hash.csv" , header = None)
Dis1 = []
Dis2 = []
Dis3 = []
Dis4 = []
Dis5 = []
Dis6 = []
with open('after_dist.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(hashcode.shape[0]):
        sevenSample = hashcode.loc[i,:]
        s1 = sevenSample[0:64]
        s2 = sevenSample[64:128]
        s3 = sevenSample[128:192]
        s4 = sevenSample[192:256]
        
        s1_array = np.array(s1)
        s2_array = np.array(s2)
        s3_array = np.array(s3)
        s4_array = np.array(s4)
        '''
        #阈值化后计算距离
        hamDist1 = np.nonzero(s1_array-s2_array)
        hamDist2 = np.nonzero(s1_array-s3_array)
        hamDist3 = np.nonzero(s1_array-s4_array)

        dist1=hamDist1[0].shape[0]
        dist2=hamDist2[0].shape[0]
        dist3=hamDist3[0].shape[0]
        
        '''
        #阈值化前计算距离
        dist1 = np.sum(np.abs(np.subtract(s1_array , s2_array)))
        dist2 = np.sum(np.abs(np.subtract(s1_array , s3_array)))
        dist3 = np.sum(np.abs(np.subtract(s1_array , s4_array)))
        
        writer.writerow([dist1,dist2,dist3])

        #Dis1.append(dist1)
        #Dis2.append(dist2)
        #Dis3.append(dist3)
        #writer.writerows(result)

    '''
    mean_dist1 = np.mean(Dis1)
    mean_dist2 = np.mean(Dis2)
    mean_dist3 = np.mean(Dis3)
    
    file=open('dist1.txt','w')  
    file.write(str(Dis1));  
    file.close()
    '''
    #print(Dis1)
