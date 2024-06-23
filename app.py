from flask import Flask, render_template, request
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader # 데이터셋 로더
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics # 지표를 보기위해 호출
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFont
import random
import os

app = Flask(__name__)

# 정적 파일 폴더를 'static' 폴더로 설정
app.config['STATIC_FOLDER'] = 'static'

class SPCNNModel(nn.Module):
    def __init__(self):
        super(SPCNNModel, self).__init__()
        # 합성곱 층
        self.num1_conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1)  # 입력된 1개의 채널을 16개로 out
        self.num2_conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)  # 아웃된 16개를 또다시 32개로 확장
        # 풀링 층
        # self.pooling = nn.MaxPool2d(kernel_size=2, stride=2) # 풀링 층을 사용하여 2분의 1크기로 다시 줄임
        # 밀집 층
        self.num1_fc = nn.Linear(32 * 5 * 5, 128)
        self.num2_fc = nn.Linear(128, 10)  # 10의 숫자 클래스
        # 활성화 함수
        # self.activation = nn.ReLu()

    def forward(self, x):  # 순전파
        # x = self.num1_conv(x) # 합성곱 층에 넣고
        # x = self.activation(x) # 활성화 함수 적용시킨 뒤
        # x = self.pooling(x) # 풀링 층에 다시 넣음
        # x = self.num2_conv(x) # 다시 한번 합성곱 층에 넣고 반복
        # x = self.activation(x)
        # x = self.pooling(x)
        x = F.max_pool2d(F.relu(self.num1_conv(x)),(2,2))
        x = F.max_pool2d(F.relu(self.num2_conv(x)),(2,2))
        x = x.view(-1,32 * 5 * 5)  # 밀집 층에 넣기전 1차원 벡터로 설정
        x = F.relu(self.num1_fc(x))
        x = self.num2_fc(x)
        return x

model = SPCNNModel()  # 객체 생성
# print(model)
model.load_state_dict(torch.load('./myTorch_mnist.pth'))
#model.eval()
protectedNumber = ""
wrong_gage = 0

mnist_transform = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize((0.5,), (1.0,))])

mnist_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)
mnist_loader = DataLoader(mnist_dataset, batch_size=128, shuffle=False, num_workers=2)

trainset = datasets.MNIST(root='/content/', train=True, download=True, transform=mnist_transform)
testset = datasets.MNIST(root='/content/', train=False, download=True, transform=mnist_transform)

train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


def train():

    model = SPCNNModel()  # 객체 생성
    # 이미지와 레이블 가져오기
    images, labels = next(iter(train_loader))

    params = list(model.parameters())
    # 손실함수
    criterion = nn.CrossEntropyLoss()
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):

        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # 역전파 추적
            optimizer.step()

            running_loss += loss.item()

            if (i % 100 == 99):
                print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            # 내가 만든 모델 저장
            PATH = './myTorch_mnist.pth'
            torch.save(model.state_dict(), PATH)


def model_load():
    # PyTorch 모델 로드 및 정의

    model = SPCNNModel()  # 객체 생성
    #print(model)
    model.load_state_dict(torch.load('./myTorch_mnist.pth'))

#app 관련 코드
@app.route('/', methods=['GET'])
def index():
    global protectedNumber
    counter = 0
    instaticImage_path = np.zeros(6, dtype='U100')

    for i in range(6):
        # 랜덤 이미지
        idx = random.randint(0, len(mnist_dataset) - 1)
        image, label = mnist_dataset[idx]

        output = model(image)
        _,predicted = torch.max(output,1)
        predicted_string = str(predicted.item())
        #print(predicted_string)

        protectedNumber += predicted_string
        #print(protectedNumber)
        # 저장경로

        image_path = f'static/{idx}.png'
        instaticImage_path[i] = f'{idx}.png'

        print(instaticImage_path[i])

        # PIL변환
        image = transforms.functional.to_pil_image(image)

        resize_image = image.resize((28 * 4, 28 * 4), resample=Image.BILINEAR)  # 이중 선형보간법 필수로 적어야함
        resize_image.save(image_path)  # 실제 데이터 저장

        # HTML 렌더링
    return render_template('index.html', image_path1 = instaticImage_path[0],
                                                          image_path2 = instaticImage_path[1],
                                                          image_path3 = instaticImage_path[2],
                                                          image_path4 = instaticImage_path[3],
                                                          image_path5 = instaticImage_path[4],
                                                          image_path6 = instaticImage_path[5],
                                                          label=label,
                                                          counter = counter)

@app.route('/renewal_num', methods=['POST'])
def renewal():
    global protectedNumber
    counter = 0
    instaticImage_path = np.zeros(6, dtype='U100')

    for i in range(6):
        # 랜덤 이미지
        idx = random.randint(0, len(mnist_dataset) - 1)
        image, label = mnist_dataset[idx]

        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_string = str(predicted.item())
        # print(predicted_string)

        protectedNumber += predicted_string
        # print(protectedNumber)
        # 저장경로

        image_path = f'static/{idx}.png'
        instaticImage_path[i] = f'{idx}.png'

        print(instaticImage_path[i])

        # PIL변환
        image = transforms.functional.to_pil_image(image)

        resize_image = image.resize((28 * 4, 28 * 4), resample=Image.BILINEAR)  # 이중 선형보간법 필수로 적어야함
        resize_image.save(image_path)  # 실제 데이터 저장

        # HTML 렌더링
    return render_template('index.html', image_path1=instaticImage_path[0],
                           image_path2=instaticImage_path[1],
                           image_path3=instaticImage_path[2],
                           image_path4=instaticImage_path[3],
                           image_path5=instaticImage_path[4],
                           image_path6=instaticImage_path[5],
                           label=label,
                           counter=counter)




@app.route('/check_input_num', methods=['POST']) # html에서 입력한 값을 post(받기)
def check_input_num():
    global protectedNumber
    global wrong_gage
    get_data = request.form['user_input_protectedNumber']
    print(get_data)
    print(protectedNumber)
    if(protectedNumber != get_data):
        wrong_gage += 1
        result_message = "잘못 입력하셨습니다."
        protectedNumber = ""
        return result_message
    result_message = "맞추셨습니다"
    protectedNumber = ""
    return result_message







if __name__ == '__main__':
    #train()
    #main()
    protectedNumber = ""
    app.run(debug=True)
