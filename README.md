# ANN을 이용한 MNIST 분류

이 프로젝트는 인공신경망(ANN, Artificial Neural Network)을 사용하여 MNIST 손글씨 데이터셋을 분류하는 모델을 구현한 예제입니다. 주어진 MNIST 데이터에 대해 ANN 모델을 학습시키고, 결과를 평가하는 방법을 설명합니다.

## 프로젝트 개요

이 코드는 MNIST 데이터셋을 사용하여 간단한 인공 신경망(ANN)을 학습하고, 손글씨 숫자 이미지를 분류하는 예제입니다. `torch.nn`을 사용하여 모델을 구성하고, 데이터를 학습시키며 정확도를 평가합니다.

## 주요 기능

- **MNIST 데이터셋 로드**: `torchvision` 라이브러리를 사용하여 MNIST 데이터셋을 다운로드하고 로드합니다.
- **ANN 모델 학습**: `torch.nn`을 사용하여 3층 신경망 모델을 정의하고 학습합니다.
- **모델 평가**: 학습된 모델을 평가하여 정확도를 출력합니다.

## 요구사항

- Python 3.x
- PyTorch
- torchvision

## 설치 방법

이 프로젝트를 실행하려면 먼저 필요한 라이브러리를 설치해야 합니다. 아래 명령어를 통해 필요한 패키지를 설치할 수 있습니다.

```bash
pip install torch torchvision
```
코드 설명
1. 데이터셋 전처리
transforms.Compose()를 사용하여 MNIST 데이터를 텐서로 변환하고, 평균 0.5와 표준편차 0.5로 정규화합니다.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```
2. MNIST 데이터셋 로드
MNIST 학습 데이터셋과 테스트 데이터셋을 로드하고, 각각 trainloader와 testloader로 데이터 로더를 설정합니다.

```python
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
```
3. ANN 모델 정의
SimpleANN 클래스는 3개의 층을 가진 간단한 신경망을 정의합니다. 입력층에서 은닉층, 은닉층에서 또 다른 은닉층으로, 마지막으로 출력층으로 연결됩니다.

```python
class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 입력층에서 은닉층으로
        self.fc2 = nn.Linear(128, 64)       # 은닉층에서 은닉층으로
        self.fc3 = nn.Linear(64, 10)        # 은닉층에서 출력층으로

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 입력 이미지를 1차원 벡터로 변환
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
4. 모델 학습
모델을 학습시키는 과정에서는 SGD 옵티마이저와 CrossEntropyLoss를 사용하여 학습을 진행합니다. 손실 값은 매 100배치마다 출력됩니다.

```python
# 손실 함수와 최적화 알고리즘 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 모델 학습
for epoch in range(10):  # 10 에포크 동안 학습
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:  # 매 100 미니배치마다 출력
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')
```
5. 모델 평가
학습 후 테스트 데이터를 이용하여 모델의 정확도를 평가합니다.

```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
```
사용법
위의 코드를 실행하여 MNIST 데이터셋을 다운로드하고 학습을 시작합니다.
학습이 완료되면, 테스트 데이터를 이용해 모델의 성능을 평가하고 정확도를 출력합니다.
