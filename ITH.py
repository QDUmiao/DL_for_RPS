import numpy as np
import ITHscore
import pandas as pd
import os

def calculate_CH_index(data, label_map):
    n, m = data.shape
    labels = np.unique(label_map)
    k = len(labels)
    
    global_center = np.mean(data)
    cluster_centers = np.array([np.mean(data[label_map == label]) for label in labels])
    within_cluster_sum_squares = sum([np.sum((data[label_map == label] - cluster_centers[i]) ** 2) for i, label in enumerate(labels)])
    between_cluster_sum_squares = sum([(np.mean(data[label_map == label]) - global_center) ** 2 * np.sum(label_map == label) for label in labels])
    
    ch_index = (between_cluster_sum_squares / (k - 1)) / (within_cluster_sum_squares / (n * m - k))
    
    return ch_index

def cal_ith(dicom_path, seg_path):
    ITH = []
    CH = []
    CLU = []
    
    image = ITHscore.load_image(dicom_path)
    seg = ITHscore.load_seg(seg_path)
    
    img, mask = ITHscore.get_largest_slice(image, seg)
    sub_img, sub_mask = ITHscore.locate_tumor(img, mask)
    features = ITHscore.extract_radiomic_features(sub_img, sub_mask, parallel=False, workers=10)
    
    print("ok")
    
    for i in range(9):
        label_map = ITHscore.pixel_clustering(sub_mask, features, cluster=i+1)
        ithscore = ITHscore.calITHscore(label_map)
        ch = calculate_CH_index(sub_img, label_map)
        
        ITH.append(ithscore)
        CH.append(ch)
        CLU.append(i+1)
    
    return CLU, ITH, CH


path = ''
names = os.listdir(path)

colname = ['id']
for i in range(9):
    colname.append('ITH_'+str(i+1))
for i in range(9):
    colname.append('CH_'+str(i+1))

csv_filename = 'extracted_data.csv'  
if os.path.exists(csv_filename):
    df = pd.read_csv(csv_filename)
else:
    df = pd.DataFrame(columns=colname)

total_names = len(names)
existing_ids = df['id'].tolist()  

for index, name in enumerate(names):
    if int(name) in existing_ids:
        print(f"Skipping {name}, already processed.")
        continue

    print(f"Processing {name} ({index + 1}/{total_names})...")
    dicom_path = os.path.join(path, name, "ori.nii.gz")
    seg_path = os.path.join(path, name, "mask.nii.gz")
    clu, ith, ch = cal_ith(dicom_path, seg_path)
    
    
    data = {'id': name}
    for i in range(9):
        data['ITH_'+str(clu[i])] = ith[i]
        data['CH_'+str(clu[i])] = ch[i]
    
    new_row = pd.DataFrame([data], columns=df.columns)
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_filename, index=False)  
    

print("All data processed and saved.")
