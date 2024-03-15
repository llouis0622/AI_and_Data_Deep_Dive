## 1. CPU vs GPU
- CPU(Central Processing Unit)
  - GPU에 비해 비교적 작은 사이즈로 제작
  - 램에서 메모리를 가져다가 사용
- GPU(Graphics Processing Unit)
  - 쿨러 따로 존재, 파워 많이 사용
  - 병렬연산 -> 행렬곱 연산 최적화 -> 딥러닝 연산 사용
  - 엔비디아 -> 딥러닝 거의 독점
  - CUDA, OpenCK, Udacity
- GPU 학습 문제
  - 모델과 가중치는 램에 상주, 실제 train data는 하드 드라이버에 존재
  - train time에 디스크에서 데이터를 잘 읽어와야 함
  - Forward/Backward는 빠름 but 디스크에서 읽어오는 것이 문제
- 해결책
  - 데이터가 작으면 램에 올리기
  - HDD 대신 SSD로 데이터 읽기 속도 개선
  - CPU 다중 스레드 사용 -> 데이터를 미리 램에 올려두기 -> 버퍼에서 GPU로 데이터 전송

## 2. Tensorflow
- Computational Graph 정의
- 그래프 실행
```
import numpy as np
import tensorflow as tf

N, D, H = 64, 1000, 100

# 그래프의 입력노드 생성, 메모리할당은 일어나지 않는다.
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
w1 = tf.placeholder(tf.float32, shape=(D, H))
w2 = tf.placeholder(tf.float32, shape=(H, D))

# x와 w1 행렬곱 연산 후 maximum을 통한 ReLU 구현
h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
diff = y_pred - y

# L2 Euclidean
loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))

# loss 계산, gradient 계산. backprop 직접구현이 필요없다
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# 아직까지 실제 계산이 이루어지지는 않았다.
# Tensorflow session : 실제 그래프를 실행
with tf.Session() as sess:
    # 그래프에 들어갈 value 지정. tensorflow는 numpy를 지원한다.
    values={x: np.random.randn(N, D),
            w1: np.random.randn(D, H),
            w2: np.random.randn(H, D),
            y: np.random.randn(N, D),}
    
    # 실제 그래프 실행. 출력으로 loss와 gradient
    # feed_dict로 실제 값 전달해주기
    # 출력 값은 numpy array
    out = sess.run([loss, grad_w1, grad_w2], feed_dict=values)
    loss_val, grad_w1_val, grad_w2_val = out
```
```
N, D, H = 64, 1000, 100

# 그래프의 입력노드 생성, 메모리할당은 일어나지 않는다.
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))

# variables로 변경. tf.random_normal로 초기화 설정
w1 = tf.Variable(tf.random_normal((D, H)))
w2 = tf.Variable(tf.random_normal((H, D)))

# x와 w1 행렬곱 연산 후 maximum을 통한 ReLU 구현
h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
diff = y_pred - y

# L2 Euclidean
loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))

# loss 계산, gradient 계산. backprop 직접구현이 필요없다
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# assign 함수를 통해 그래프 내에서 업뎃이 일어날 수 있도록 해줌
learning_rate = 1e-5
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# w1와 w2를 업뎃하라고 명시적으로 넣어주어야 한다.
# tensorflow는 output에 필요한 연산만 수행한다.
# new_w1, new_w2를 직접적으로 넣어줄 수 있으나, 사이즈가 큰 tensor의 경우
# tensorflow가 출력을 하는 것은 cpu/gpu간 데이터 전송이 필요하므로 좋지 않다

# 따라서 dummy node인 updates를 만들어 그래프에 추가
updates = tf.group(new_w1, new_w2)

# 아직까지 실제 계산이 이루어지지는 않았다.

# Tensorflow session: 실제 그래프를 실행
with tf.Session() as sess:
    # 그래프 내부 변수들 초기화
    sess.run(tf.global_variables_initializer())
    
    values = {x: np.random.randn(N, D), y: np.random.randn(N, D),}
    for t in range(50):
        loss_val = sess.run([loss, updates], feed_dict=values)
```
```
N, D, H = 64, 1000, 100

# 그래프의 입력노드 생성, 메모리할당은 일어나지 않는다.
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))

# Xavier로 초기화
init = tf.contrib.layers.xavier_initializer()
# 내부적으로 w1, b2를 변수로 만들어주고 초기화
h = tf.layers.dense(inputs=x, units=H, activation=tf.nn.relu, kernel_initializer=init)
y_pred = tf.layers.dense(inputs=h, units=D, kernel_initializer=init)

# tensorflow 내장함수로 변경가능
loss = tf.losses.mean_squared_error(y_pred, y)

# optimizer을 이용해서 gradient를 계산하고 가중치를 업뎃할 수 있다
optimizer = tf.train.GradientDescentOptimizer(1e-5)
updates = optimizer.minimize(loss)

# 아직까지 실제 계산이 이루어지지는 않았다.

# Tensorflow session: 실제 그래프를 실행
with tf.Session() as sess:
    # 그래프 내부 변수들 초기화
    sess.run(tf.global_variables_initializer())
    
    values={x: np.random.randn(N, D), y: np.random.randn(N, D),}
    for t in range(50):
        loss_val = sess.run([loss, updates], feed_dict=values)
```
- Tensorflow 기반 High Level Wrapper
  - Keras
  - TFLearn
  - TensorLayer
  - tf.layers
  - TF-Slim
  - tf.contrib.learn
  - Pretty Tensor

## 3. Pytorch
- tensor : Imperative Array, GPU에서 돌아감
- variable : 그래프 노드, 그래프 구성 및 그레이디언트 계산
- module : NN 구성
```
import torch
from torch.autograd import Variable

# GPU에서 돌아가도록 데이터 타입을 변경
dtype = torch.cuda.FloatTensor

N, D_in, H, D_out = 64, 1000, 100, 10

# 가중치에 대한 gradient만 True로 변경한다
x = Variable(torch.randn(N, D_in), requires_grad=False)
y = Variable(torch.randn(N, D_out), requires_grad=False)
w1 = Variable(torch.randn(D_in, H), requires_grad=True)
w2 = Variable(torch.randn(H, D_out), requires_grad=True)

#학습률 설정
learning_rate = 1e-6

for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    
    if w1.grad: w1.grad.data.zero_()
    if w2.grad: w2.grad.data.zero_()
    loss.backward()
    
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
```
```
import torch
from torch.autograd import Variable

# GPU에서 돌아가도록 데이터 타입을 변경
dtype = torch.cuda.FloatTensor

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Define our model as a sequence of layers
model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out))

# common loss functions도 제공한다
loss_fn = torch.MSELoss(size_average=False)

#학습률 설정
learning_rate = 1e-4

for t in range(500):
    # model에 x 넣고 prediction
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    
    #backward pass, gradient 계산
    model.zero_grad()
    loss.backward()
    
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data
```
```
import torch
from torch.autograd import Variable

# GPU에서 돌아가도록 데이터 타입을 변경
dtype = torch.cuda.FloatTensor

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Define our model as a sequence of layers
model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out))

# common loss functions도 제공한다
loss_fn = torch.MSELoss(size_average=False)

# 학습률 설정
learning_rate = 1e-4

# Use optimizer for different update rules
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    # model에 x 넣고 prediction
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    
    # backward pass, gradient 계산
    optimizer.zero_grad()
    loss.backward()
    
    # 그레디언트 계산 후 모든 파라미터 업뎃
    optimizer.step()
```
```
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDatasetm DataLoader

# 단일 모듈로 model 생성
# backward는 autograd가 알아서 해줌
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1=torch.nn.Linear(D_in, H)
        self.linear2=torch.nn.Linear(H, D_out)
        
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# DataLoader가 minibatching, shuffling, multithreading 관리
loader = DataLoader(TensorDataset(x, y).batch_size=8)

model = TwoLayerNet(D_in, H, D_out)
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    for x_batch, y_batch in loader:
        # model에 x 넣고 prediction
        x_var, y_var = Variable(x), Variable(y)
        y_pred = model(x_var)
        loss = criterion(y_pred, y_var)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 4. Tensorflow vs Pytorch
- Static(Tensorflow)
  - 그래프 하나 고정
  - 해당 그래프 최적화 가능
  - 원본 코드 없이도 그래프 다시 불러오기 가능
  - 조건부 연산 그래프 따로 제작
- Dynamic(Pytorch)
  - Forward Pass 할 때마다 새로운 그래프 구성
  - 최적화 어려움
  - 모델 재사용을 위해 항상 원본 코드 필요
  - 깔끔한 코드와 작성 우수
  - RNN에 요긴하게 사용

## 5. Etc
- Caffe
  - 코드 작성 없이도 네트워크 학습 가능
  - Feed Forward 모델 적합
  - Prodiction 측면 적합
- Caffe2
  - Static Graph 지원
  - 코어 -> C++, Python 인터페이스 제공
- Google
  - 딥러닝이 필요한 모든 곳에서 동작하는 프레임워크 제작 원함
- Facebook
  - Pytorch 연구 특화
  - caffe2 제품개발 사용