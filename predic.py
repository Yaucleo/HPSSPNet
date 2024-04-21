# predict.py
# import time
from time import time

import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, plot_confusion_matrix, \
    roc_auc_score
from torch.nn import Linear
# from Alexnet_model import AlexNet
# from sklearn.metrics import accuracy_score, f1_score
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import os
# from tqdm import tqdm
# import pandas as pd
# from model import *
from torchvision import transforms, datasets, utils
from evaluation import outputsforaoc
from ZF_Net import ZFNet
from oursresnet import resnet18ours
from evaluation import *
from torchvision import transforms, datasets

from oursresnet import resnet18ours

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def predict(net, device, k, batch_size = 64):
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_dataset = datasets.ImageFolder(root='testdata7.11',
                                         transform=data_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=batch_size, shuffle=True,
                                                       num_workers=0)
    # load model weights
    model_weight_path = "new_result/zfnet/" + str(k) + '.pth'
    # Resnet densenet Alexnet vgg16
    net.load_state_dict(torch.load(model_weight_path))

    # 二分类交叉熵
    # loss_function = nn.BCEWithLogitsLoss()
    # 参数
    frame = 0
    acces = []
    f1es = []
    losses = []
    fpres = []
    tpres = []
    labels_forroc = []
    predict_forroc = []
    acc_pre =[]
    acc_lable = []
    net.eval()
    acc = 0.0
    f1 = 0.0
    # 画图
    acces = []
    f1es = []

    # with torch.no_grad():
    net.eval()

    pre = []
    true = []
    outputforaoc = []
    t1 = time()
    for test_data in test_loader:
        frame += 1
        test_images, test_labels = test_data
        outputs = net(test_images.to(device))
        predict_y = torch.max(outputs, dim=1)[1]

        outputforaoc.extend(outputsforaoc(outputs))

        pre.extend(predict_y.cpu().numpy())
        true.extend(test_labels.cpu().numpy())
        # acc += (predict_y == val_labels.to(device)).sum().item()
        # f1 += f1_score(val_labels.cpu().numpy(), predict_y.cpu().numpy())
    t2 = time()
    # 消耗时间
    print(t2-t1)
    test_accurate = accuracy_score(true, pre)
    test_f1 = f1_score(true, pre)
    # 召回率
    test_recall = recall_score(true, pre)
    # 准确率
    test_precision = precision_score(true, pre)


    acces_all.append(test_accurate)
    f1es_all.append(test_f1)
    recalles_all.append(test_recall)
    precisones_all.append(test_precision)
    fps_all.append(840/(t2 - t1))
    print("acc:{:.4f}, f1:{:.4f}, recall:{:.4f}, precison:{:.4f}".format(test_accurate, test_f1, test_recall, test_precision))

        # # 混淆矩阵可视化
        # from sklearn.metrics import confusion_matrix
        # from sklearn.metrics import ConfusionMatrixDisplay
        #
        # confusion_mat = confusion_matrix(true, pre)
        #
        # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
        # disp.plot(
        #     include_values=True,  # 混淆矩阵每个单元格上显示具体数值
        #     cmap="viridis",  # 不清楚啥意思，没研究，使用的sklearn中的默认值
        #     ax=None,  # 同上
        #     xticks_rotation="horizontal",  # 同上
        #     values_format="d"  # 显示的数值格式
        # )
        # # plt.show()
        # # cm_display = ConfusionMatrixDisplay(confusion_mat).plot()
        #
        # # ROC曲线可视化
        # from sklearn.metrics import roc_curve
        # from sklearn.metrics import RocCurveDisplay
        # # y_score = clf.decision_function(X_test)
        #
        # fpr, tpr, _ = roc_curve(true, outputforaoc, pos_label=1)
        # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
        # roc_display.plot(
        #     label='AUC:{:.3f}'.format(roc_auc_score(true, pre))
        # )
        # plt.show()
        # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

        # val_accurate = acc / val_num
        # val_f1 = f1 / val_num
        # # f1_score =
        # cross_acces.append(val_accurate)
        # cross_f1es.append(val_f1)
        # # try:
        # with tqdm(test_loader) as t:
        #     for data in t:
        #         # 将数据分为图像和标签
        #         test_images, test_labels = data
        #         # labels = labelchange(test_labels)
        #         # labels = labels.to(device=device)
        #         test_images = test_images.to(device=device)
        #         test_labels = test_labels.to(device=device)
        #         outputs = net(test_images)
        #         outputs = outputs.squeeze(1)
        #         test_labels = test_labels.float()
        #         loss = loss_function(outputs, test_labels)
        #         losses.append(loss.item())
        #         outputs = torch.sigmoid(outputs)

                # 用于roc曲线的绘制
    #             labels_forroc += labelsfordraw(test_labels)
    #             predict_forroc += outputsfordraw(outputs)
    #
    #
    #             # m: 0 => column  1 => row
    #             predict_y = sigmoid_pre(outputs)
    #             # print(predict_y.item())
    #             acc_pre = acc_pre + predict_y.cpu().tolist()
    #             acc_lable = acc_lable + test_labels.cpu().tolist()
    #             #       将预测与真实标签进行对比，对的为1，错的为0进行求和--》求出正确的样本个数
    #             # acc = acc_compute(predict_y, test_labels, device, batch_size)
    #             acces.append(acc_compute(predict_y, test_labels, device, batch_size))
    #             f1es.append(f1_compute(test_labels, predict_y))
    #
    #             # test_tpr, test_fpr, N, P = ROC(test_labels, predict_y)
    #             #
    #             # # 当全都是sp时没有负样本不计算
    #             # if N != 0 and P != 0:
    #             #     # 用于输出
    #             #     fpres.append(test_fpr)
    #             #     tpres.append(test_tpr)
    #
    #     # except KeyboardInterrupt:
    #     #     t.close()
    #     # t.close()
    # loss = np.average(losses)
    # acc = np.average(acces)
    # f1 = np.average(f1es)
    # fpr = np.average(fpres)
    # tpr = np.average(tpres)
    #
    # # print_msg = (
    # #              f'loss: {loss:.5f} ' +
    # #              f'acc: {acc:.5f} ' +
    # #              f'f1: {f1:.5f} ' +
    # #              f'tpr: {tpr:.5f} ' +
    # #              f'fpr: {fpr:.5f} '
    # #              )
    # TPR, FPR = TPRandFPR(acc_lable, acc_pre)
    # # print(print_msg)
    # print('tpr: %.5f' % TPR)
    # print('fpr: %.5f' % FPR)
    # print('acc: %.5f' % accuracy_score(acc_lable, acc_pre))
    # print('f1: %.5f' % f1_score(acc_lable, acc_pre))
    # draw_roc(labels_forroc, predict_forroc)
    # # ld image
    # test_path = os.getcwd() + "/test2/rgb"  # flower data set path
    # test_files = os.listdir(test_path)
    # # load image
    #
    #     # print(file)
    #
    # # # read class_indict
    # try:
    #     json_file = open('class_indices.json', 'r')
    #     class_indict = json.load(json_file)
    # except Exception as e:
    #     print(e)
    #     exit(-1)
    #
    # # create model
    # model = torchvision.models.mobilenet_v2(pretrained=False)
    # model.classifier[1] = Linear(1280, 2)
    # model.load_state_dict(torch.load('mobilenet.pth'), strict=False)
    # # model = AlexNet(num_classes=2, init_weights=True)
    # # model = pretrained_ResNet(num_classes=2)
    # # load model weights
    # # model_weight_path = "AlexNet07.pth"
    # # model.load_state_dict(torch.load(model_weight_path))
    # model.eval()
    # id_list = []
    # pred_list = []
    # with torch.no_grad():
    #     # predict class
    #     for file in tqdm(test_files):
    #         img = Image.open(test_path + '/' +file)  # 验证太阳花
    #         # img = Image.open("roses.jpg")     #验证玫瑰花
    #         # img.convert('RGB')
    #         # [N, C, H, W]
    #         img = data_transform(img)
    #         # expand batch dimension
    #         img = torch.unsqueeze(img, dim=0)
    #         output = torch.squeeze(model(img))
    #         predict = torch.softmax(output, dim=0)
    #         predict_cla = torch.argmax(predict).numpy()
    #         id_list.append((file.split('.')[0]).split('_')[5])
    #         pred_list.append(predict_cla)
    #         # print(predict_cla)
    #
    # # for cla in range(len(class_indict)):
    # #     print(class_indict[str(cla)] ,':', predict[cla].numpy())
    # # print(class_indict[str(predict_cla)],'!')
    # data = {
    #     "lables": id_list,
    #     "class": pred_list
    # }
    #
    # df = pd.DataFrame(data)
    # # csv = df.to_csv(test_path + '/test.csv')
    # print(df)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # net = resnet34(num_classes=2)
    # net = torchvision.models.squeezenet1_0(num_classes=2)
    # print(net)

    # net = resnet18ours(num_classes=2,include_top=True)
    # print(net)
    acces_all = []
    f1es_all = []
    recalles_all = []
    precisones_all = []
    fps_all = []
    for i in range(1, 6):
        # net = torchvision.models.resnet18(num_classes=2)
        # net = resnet18ours()
        net = ZFNet(num_classes=2)
    # 网络指定到规定的设备中
        net.to(device)
        predict(net, device, i)
    print('----------------------------------------------------')
    print("acc:{:.4f}, f1:{:.4f}, recall:{:.4f}, precison:{:.4f}, fps:{:.4f}".format(np.mean(acces_all), np.mean(f1es_all), np.mean(recalles_all), np.mean(precisones_all), np.mean(fps_all)))
    print("acc:{:.4f}, f1:{:.4f}, recall:{:.4f}, precison:{:.4f}, fps:{:.4f}".format(np.std(acces_all), np.std(f1es_all), np.std(recalles_all), np.std(precisones_all), np.std(fps_all)))







