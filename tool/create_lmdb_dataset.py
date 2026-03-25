import os
import lmdb
import cv2
import numpy as np

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode('utf-8'), v)

def createDataset(outputPath, imagePathList, labelList, checkValid=True):
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    
    # 1TB map size, đủ sức chứa dataset lớn
    env = lmdb.open(outputPath, map_size=1099511627776)
    
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
            
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
            
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
        
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples at %s' % (nSamples, outputPath))

def read_txt(txt_path, img_dir):
    imagePathList = []
    labelList = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # Format thường gặp: "đường_dẫn_ảnh <tab> nhãn" hoặc "đường_dẫn_ảnh <space> nhãn"
            parts = line.split('\t')
            if len(parts) != 2:
                # Fallback to space split if tab is not found
                parts = line.split(' ', 1)
            
            if len(parts) == 2:
                img_path, label = parts[0], parts[1]
                # Lắp thành đường dẫn tuyệt đối
                full_path = os.path.join(img_dir, img_path)
                imagePathList.append(full_path)
                labelList.append(label)
    return imagePathList, labelList

if __name__ == '__main__':
    # 1. Chuyển đổi tập Train
    print("Đang tạo LMDB cho tập Train...")
    train_img_list, train_label_list = read_txt('rec/rec_gt_train.txt', 'rec')
    # Tạo thư mục đầu ra
    os.makedirs('lmdb_dataset/train', exist_ok=True)
    createDataset('lmdb_dataset/train', train_img_list, train_label_list)
    
    # 2. Chuyển đổi tập Val
    print("\nĐang tạo LMDB cho tập Val...")
    val_img_list, val_label_list = read_txt('rec/rec_gt_val.txt', 'rec')
    os.makedirs('lmdb_dataset/val', exist_ok=True)
    createDataset('lmdb_dataset/val', val_img_list, val_label_list)
    
    print("\nHoàn tất! Bây giờ bạn có thể trỏ --trainRoot tới 'lmdb_dataset/train' và --valRoot tới 'lmdb_dataset/val'")
