from cProfile import label
import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.cm as cm
import pickle


#from logging import getLogger


# 平均と現在の値を計算
class AverageMeter(object):
    def __init__(self, name):
        self.reset()
        self.name = name

    @property
    def avg(self):
        return self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

#学習曲線を描いたり
class History(object):

    def __init__(self, keys, output_dir):

        self.output_dir = output_dir
        self.keys = keys

        self.logs = {key: [] for key in keys}

        col = ["color","healthy","satisfaction","uniqueness","ease of eating","appropriate amount","not collapse"]

        self.col = col
    
    # 引数：data　historyが呼ばれたときに辞書型のkeyとvalueをlogsに追加する
    def __call__(self, data): 
        for key, value in data.items():
            self.logs[key].append(value)

    def save(self, filename='history.pkl'):
        savepath = os.path.join(self.output_dir, filename)
        with open(savepath, 'wb') as f:
            pickle.dump(self.logs, f)

        with open(os.path.join(self.output_dir, "histry.csv"), 'a') as f: # 'a' 追記
            writer = csv.writer(f)
            for key, value in self.logs.items():
                writer.writerow([key,value])


    def plot_loss(self, keylist, filename='loss_.png'):
        plt.figure(figsize=(8, 4.5))
        for key in keylist:
            plt.plot(self.logs[key])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        #plt.ylim(0, 50000)
        plt.legend(keylist, loc='upper right')
        plt.grid(True)
        plt.show()

        save_path = os.path.join(self.output_dir, filename)
        #logger.info('Save {}'.format(save_path))
        plt.savefig(save_path, transparent=True)
        plt.clf() # 図全体をクリア
        plt.cla() # 軸をクリア
        plt.close('all') # closes all the figure windows


    def plot_roc_curve(self, filename='roc.png'):
        fpr = self.logs['fpr']
        tpr = self.logs['tpr']
        plt.plot(fpr, tpr, marker='o')
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.grid()

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, transparent=True)
        plt.clf() # 図全体をクリア
        plt.cla() # 軸をクリア
        plt.close('all') # closes all the figure windows


    def plot_acc(self, filename='acc.png'):
        train_x = np.arange(len(self.logs['train_acc']))
        train_acc = self.logs['train_acc']

        val_x = np.arange(len(self.logs['val_acc']))
        val_acc = self.logs['val_acc']


        plt.title("acc")
        plt.plot(train_x, train_acc, color='blue', label='train_acc')
        plt.plot(val_x, val_acc, color='red', label='val_acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend(loc='best')

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, transparent=True)
        plt.clf() # 図全体をクリア
        plt.cla() # 軸をクリア
        plt.close('all') # closes all the figure windows

    # 入力はバッチごとのリストで来る
    def radar_chart(self ,input_img, output, score, dataframe , index_list, save_path, filename='radar_chart.png'):
        img_path = dataframe['path']

        #col = ["彩り", "健康的", "満足感", "ユニークさ", "食べやすさ", "適量", "くずれない"]
        col = self.col

        for i in range(len(output)):
            L1_loss = abs(output[i] - score[i])

            # 多角形を閉じるためにデータの最後に最初の値を追加する。
            output_values = np.append(output[i], output[i][0])
            score_values  = np.append(score[i], score[i][0])

            # プロットする角度を生成する。
            angles = np.linspace(0, 2 * np.pi, len(col) + 1 , endpoint=True)

            fig = plt.figure(figsize=(12, 12))

            # libのmatplotのファイルのデフォルトフォントを変えた
            #plt.rcParams['font.family'] = 'IPAexGothic'
            #print(plt.rcParams["font.family"])
            #plt.rcParams["font.family"] = 'sans-serif'   # 使用するフォント
            ax0 = fig.add_subplot(2, 2, 1)
            ax1 = fig.add_subplot(2, 2, 2, polar=True)

            #img_path = data_dir +  img_name[index_list[i]]
            #ax0.imshow(mpimg.imread(img_path))

            ax0.imshow(input_img[i].transpose(1, 2, 0))
            ax0.set_title(img_path[index_list[i]], pad=20)


            # 極座標でaxを作成。
            # レーダーチャートの線を引く
            ax1.plot(angles, output_values, label="output")
            ax1.plot(angles, score_values, label="correct data")

            # 項目ラベルの表示
            ax1.set_thetagrids(angles[:-1] * 180 / np.pi, col)#,fontname="IPAexGothic"
            ax1.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0]) # メモリ線
            
            ax1.legend(bbox_to_anchor=(1, 1), loc='center', borderaxespad=0)

            ax1.set_title(f"L1loss_mean:{np.round(L1_loss.mean(), 3)}" , pad=20)
            plt.show()

            img_path = save_path +  f"/{i}_"+filename

            plt.savefig(img_path)#, transparent=True)
            plt.clf() # 図全体をクリア
            plt.cla() # 軸をクリア
            plt.close('all') # closes all the figure windows


    # yyplot 作成関数
    def score_pred_plot(self, score, output, filename='score_predicted.png'):
        #output = output.to('cpu').detach().numpy().copy()
        #score  = score.to('cpu').detach().numpy().copy()
        
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
        #col = ["彩り", "健康的", "満足感", "ユニークさ", "食べやすさ", "適量", "くずれない"]
        ax = ax.flatten()

        #plt.rcParams["font.family"] = 'sans-serif'   # 使用するフォント
        col = self.col

        for i in range(1,len(ax)):

            ax[i].scatter(  score[:,i-1], 
                            output[:,i-1], 
                            label=col[i-1] + f":{np.corrcoef(score[:,i-1],output[:,i-1])[0,1]}")
            #ax[i].set_xlim(0,1)
            #ax[i].set_ylim(0,1)
            ax[i].set_xlabel('score', fontsize=14)
            ax[i].set_ylabel('output', fontsize=14)
            ax[i].legend(loc="upper right")


        save_path = os.path.join(self.output_dir, filename)
        
        plt.savefig(save_path, transparent=True)
        plt.clf() # 図全体をクリア
        plt.cla() # 軸をクリア
        plt.close('all') # closes all the figure windows
