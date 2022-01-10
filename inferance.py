import torch
import torch.nn as nn
import pandas as pd
import os, cv2, re, random, time


# 调用保存的最佳模型的准确率输出
def natural_keys(text):
    '''输入字符串，将数字与文字分隔开，将数字串转化为int'''
    return [ txt_dig(c) for c in re.split('(\d+)', text) ]
	
def main():
    #dataset = Food_LT_wyg(False, root=cfg.root, batch_size=cfg.batch_size, num_works=4)
    #test_loader = dataset.test_instance#加载训练数据
    TEST_DIR = './data/food/test'
	test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)]
	test_images.sort(key=natural_keys)
	#预处理
	test = []
    for img in test_images:
        test.append(cv2.resize(cv2.imread(img), 
                        (IMG_WIDTH, IMG_HEIGHT), 
                        interpolation=cv2.INTER_CUBIC))
	print('The shape of test data is {}'.format(np.array(test).shape))					
	#加载模型
    resume = '/ckpt/model_best.pth.tar'
    model_test = torch.load(resume)#加载用训练数据训练好的模型
	#测试集测试
	test = test.astype('float32') / 255 # 归一化
    test_pred = model_test.predict(test)
	submission = pd.DataFrame({'id': range(1, len(test_images) + 1), 'label': test_pred.ravel()})
    submission.to_csv('./output/submission.csv', index = False)
    print('This program costs {:.2f} seconds'.format(time.time()-start))
    print('test Finish !')
	
if __name__ == '__main__':
    main()