import numpy as np
import matplotlib.pyplot as plt
import ITHscore

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

def cal_ith(dicom_path,seg_path):
    ITH = []
    CH = []
    CLU = []
    
    image = ITHscore.load_image(dicom_path)
    seg = ITHscore.load_seg(seg_path)
    
    img, mask = ITHscore.get_largest_slice(image, seg)
    
    sub_img, sub_mask = ITHscore.locate_tumor(img, mask)
    
    
    features = ITHscore.extract_radiomic_features(sub_img, sub_mask, parallel=False,workers=10)
    
    for i in range(9):
        label_map = ITHscore.pixel_clustering(sub_mask, features, cluster=i+1)
        ithscore = ITHscore.calITHscore(label_map)
        ch = calculate_CH_index(sub_img,label_map)
        
        print("i")
        print("ith:",ithscore)
        print("ch:",ch)
        
        ITH.append(ithscore)
        CH.append(ch)
        CLU.append(i)
    
    return CLU,ITH,CH
        
    
dicom_path = ""
seg_path = ""

image = ITHscore.load_image(dicom_path)
seg = ITHscore.load_seg(seg_path)
print(image.shape, seg.shape)

img, mask = ITHscore.get_largest_slice(image, seg)

plt.subplot(131)
plt.imshow(img, cmap="bone")
plt.title("Image")
plt.subplot(132)
plt.imshow(mask, cmap="gray")
plt.title("Mask")
plt.subplot(133)
plt.imshow(img, cmap="bone")
plt.imshow(mask, alpha=0.5)
plt.title("Stack")




sub_img, sub_mask = ITHscore.locate_tumor(img, mask)
plt.subplot(121)
plt.imshow(sub_img, cmap="bone")
plt.title("Tumor")
plt.subplot(122)
plt.imshow(sub_img, cmap="bone")
plt.imshow(sub_mask, alpha=0.5)
plt.title("Stack")

features = ITHscore.extract_radiomic_features(sub_img, sub_mask, parallel=False,workers=20)

label_map = ITHscore.pixel_clustering(sub_mask, features, cluster=5)
fig = ITHscore.visualize(img, sub_img, mask, sub_mask, features, cluster="all")
ithscore = ITHscore.calITHscore(label_map)
ch = calculate_CH_index(sub_img,label_map)
print("ith:",ithscore)
print("ch:",ch)

label_map = ITHscore.pixel_clustering(sub_mask, features, cluster=6)
# fig = ITHscore.visualize(img, sub_img, mask, sub_mask, features, cluster="all")
ithscore = ITHscore.calITHscore(label_map)
ch = calculate_CH_index(sub_img,label_map)
print("ith:",ithscore)
print("ch:",ch)

label_map = ITHscore.pixel_clustering(sub_mask, features, cluster=7)
# fig = ITHscore.visualize(img, sub_img, mask, sub_mask, features, cluster="all")
ithscore = ITHscore.calITHscore(label_map)
ch = calculate_CH_index(sub_img,label_map)
print("ith:",ithscore)
print("ch:",ch)













from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
def pixel_clustering(sub_mask, features, cluster=6):
    """
    Args:
        sub_mask: Numpy array. ROI mask only within the tumor bounding box
        features: Numpy array or dict (output of previous step). Matrix of radiomic features. Rows are pixels and columns are features
        cluster: Int. The cluster number in clustering
    Returns:
        label_map: Numpy array. Labels of pixels within tumor. Same size as tumor_img
    """
    if isinstance(features, dict):
        features = np.hstack((features['first'], features['shape'], features['glcm'], features['gldm'],
                              features['glrlm'], features['glszm'], features['ngtdm']))
    features = MinMaxScaler().fit_transform(features)
    label_map = sub_mask.copy()
    clusters = KMeans(n_clusters=cluster).fit_predict(features)
    cnt = 0
    for i in range(len(sub_mask)):
        for j in range(len(sub_mask[0])):
            if sub_mask[i][j] == 1:
                label_map[i][j] = clusters[cnt] + 1
                cnt += 1
            else:
                label_map[i][j] = 0

    return label_map


def visualize2(img, sub_img, mask, sub_mask, features, cluster=6):  
    """  
    Args:  
        img: Numpy array. Original whole image, used for display  
        sub_img: Numpy array. Tumor image  
        mask: Numpy array. Same size as img, 1 for tumor and 0 for background, used for display  
        sub_mask: Numpy array. Same size as sub_img, 1 for nodule and 0 for background  
        features: Numpy array. Matrix of radiomic features. Rows are pixels and columns are features  
        cluster: Int or Str. Integer defines the cluster number in clustering. "all" means iterate clusters from 2 to 9 to generate multiple cluster pattern.  
    Returns:  
        fig: figure for display  
    """  
    if cluster != "all":  
        if not isinstance(cluster, int):  
            raise Exception("Please input an integer or string 'all'")  
        fig = plt.figure()  
        label_map = pixel_clustering(sub_mask, features, cluster)  
        plt.matshow(label_map, fignum=0)  
        plt.xlabel(f"Cluster pattern (K={cluster})", fontsize=15)  

        return fig  

    else:  # generate cluster pattern with multiple resolutions, together with whole lung CT  
        max_cluster = 8
        min_cluster = 2  
        # Subplot 1: CT image of the whole lung  
        fig = plt.figure(figsize=(10, 10))  
        plt.subplot(3, (max_cluster + 1) // 3, 1)  
        plt.title('Raw Image')  
        plt.imshow(img, cmap='gray')  
        plt.scatter(np.where(mask == 1)[1], np.where(mask == 1)[0], marker='o', color='red', s=0.2)  

        # Subplot 2: CT image of the nodule  
        plt.subplot(3, (max_cluster + 1) // 3, 2)  
        plt.title('Tumor')  
        plt.imshow(sub_img, cmap='gray')  

        # Subplot 3~n: cluster label map with different K  
        area = np.sum(sub_mask == 1)  
        for clu in range(min_cluster, max_cluster + 1):  
            label_map = pixel_clustering(sub_mask, features, clu)  
            plt.subplot(3, (max_cluster + 1) // 3, clu + 1)  # Adjust index to start from subplot 3  
            plt.matshow(label_map, fignum=0)  
            plt.xlabel(str(clu) + ' clusters', fontsize=15)  
        plt.subplots_adjust(hspace=0.3)  
        plt.suptitle(f'Cluster pattern with multiple resolutions (area = {area})', fontsize=15)  

        return fig  
    
    
    
    
fig = visualize2(img, sub_img, mask, sub_mask, features, cluster="all")
