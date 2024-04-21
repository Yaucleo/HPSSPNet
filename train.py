import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
# from Alexnet_model import AlexNet
from numpy import *
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from LeNet import LeNet
from ZF_Net import ZFNet
from oursresnet import resnet18ours
# import EarlyStopping
from pytorchtools import EarlyStopping

def main(net, batch_size=64, epochs = 200):
    writer = SummaryWriter("logs")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪到224×224
                                     # transforms.CenterCrop(224),
                                     # 水平方向上随机翻转
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     # 标准化处理
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224,must (224,224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    image_path = os.getcwd() +"/dataset/"
    train_dataset = datasets.ImageFolder(root=image_path + "/1" + "/train",
                                         transform=data_transform["train"])
    #
    # validate_dataset = datasets.ImageFolder(root=image_path + "/val",
    #                                         transform=data_transform["val"])

    # train_num = len(train_dataset)

    # flower_list={'daisy':0,'dandelion':1,'roses':2,'sunflower':3,'tulips':4}
    flower_list = train_dataset.class_to_idx
    # 遍历flower_list字典,将key和value反过来
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)


    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=batch_size, shuffle=True,
    #                                            num_workers=0)
    #
    # val_num = len(validate_dataset)
    # validate_loader = torch.utils.data.DataLoader(validate_dataset,
    #                                               batch_size=4, shuffle=False,
    #                                               num_workers=0)

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()



    # net = AlexNet(num_classes=2, init_weights=True)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    # pata=list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # k折
    k_flod = 0
    save_path = './alexnet.pth'
    best_acc = 0.0
    best_f1 = 0.0
    cross_losses = []
    cross_acces = []
    cross_f1es = []
    val_loss = []

    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 20

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # 五折保存loss最小的模型
    smallest_loss = inf
    best_weight = 0
    for epoch in range(epochs):
        # 数据集
        k_flod += 1
        train_dataset = datasets.ImageFolder(root=image_path + "/" +str(k_flod) + "/train",
                                             transform=data_transform["train"])

        validate_dataset = datasets.ImageFolder(root=image_path + "/" +str(k_flod)+ "/val",
                                                transform=data_transform["val"])

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=0)

        val_num = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=batch_size, shuffle=False,
                                                      num_workers=0)



        # train,管理dropout方法
        net.train()
        running_loss = 0.0
        # 训练一个epoch所需要的时间
        t1 = time.perf_counter()
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print train process
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        print()
        print(time.perf_counter() - t1)

        # validate

        net.eval()
        acc = 0.0  # accumulate accurate number /epoch
        f1 = 0.0
        with torch.no_grad():
            pre = []
            true = []
            # smallest_loss = inf
            # best_weight = 0
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # for early stopping
                loss = loss_function(outputs, val_labels.to(device))


                val_loss.append(loss.item())

                predict_y = torch.max(outputs, dim=1)[1]
                pre.extend(predict_y.cpu().numpy())
                true.extend(val_labels.cpu().numpy())

                # acc += (predict_y == val_labels.to(device)).sum().item()
                # f1 += f1_score(val_labels.cpu().numpy(), predict_y.cpu().numpy())
            val_accurate = accuracy_score(true, pre)
            val_f1 = f1_score(true, pre)
            # val_accurate = acc / val_num
            # val_f1 = f1 / val_num
            # f1_score =
            cross_acces.append(val_accurate)
            cross_f1es.append(val_f1)

            # 保存loss最小的模型
            if mean(val_loss) < smallest_loss:
                best_weight = net.state_dict()
                smallest_loss = mean(val_loss)
                print('The best weight is', k_flod)

            # early_stopping needs the validation loss to check if it has decresed,
            optimizer.param_groups[0]["lr"] *= 0.9
            # and if it has, it will make a checkpoint of the current model
            early_stopping(mean(val_loss), net)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if k_flod % 5 == 0:
                if mean(cross_f1es) > best_f1:
                    best_f1 = mean(cross_f1es)
                    best_acc = mean(cross_acces)
                    torch.save(best_weight, save_path)
                    print('Save successfully !!!')

                cross_acces =[]
                cross_f1es = []
                smallest_loss = inf
                best_weight = 0

            # 写入
            writer.add_scalar('train_loss', running_loss / step, epoch)
            writer.add_scalar('val_loss', mean(val_loss), epoch)
            print('[epoch %d] train_loss: %.3f  val_loss: %.3f val_accuracy: %.3f val_f1: %.3f' %
                (epoch + 1, running_loss / step, mean(val_loss), val_accurate, val_f1))

            val_loss = []

        if k_flod == 5:
            k_flod = 0
    writer.close()
    print('Finished Training')


if __name__ == '__main__':
    # net = torchvision.models.shufflenet_v2_x1_0(num_classes=2)
    # net = torchvision.models.alexnet(num_classes = 2)
    # models.shufflenet_v2_x1_0(pretrained=True)
    net = resnet18ours(num_classes=2,include_top=True)
    # print(net)
    # net = LeNet()
    # net = AlexNet(num_classes=2, init_weights=True)
    main(net)